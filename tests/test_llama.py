"""CPU-only regression tests: no NVIDIA queries or CUDA allocations."""
import argparse
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

ENTRY = Path(__file__).resolve().parents[1] / 'llama'
loader = importlib.machinery.SourceFileLoader('gpu_hold', str(ENTRY))
spec = importlib.util.spec_from_loader(loader.name, loader)
gpu = importlib.util.module_from_spec(spec)
loader.exec_module(gpu)


class Tests(unittest.TestCase):
    def test_memory_validation(self):
        self.assertEqual(gpu.memory_size('0.5'), .5)
        for value in ('0', '-1', 'nan', 'inf', '0.00000000001'):
            with self.assertRaises(argparse.ArgumentTypeError):
                gpu.memory_size(value)

    def test_cli_rejects_invalid_modes_without_cuda(self):
        for args in ([], ['--memory', '2'], ['--release', '0'], ['--pid', '1'],
                     ['--memory', '2', '--release', '0', '--pid', '1']):
            result = subprocess.run([sys.executable, '-B', str(ENTRY), *args],
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_config_validation(self):
        gpu.validate_config()
        for key, value in [('CLAIM_MODE', 'bad'), ('COMPUTE_HOURS', 0),
                           ('POLL_SECONDS', float('nan')), ('BLOCK_MIB', .5)]:
            with patch.object(gpu, key, value), self.assertRaises(ValueError):
                gpu.validate_config()

    def test_release_real_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            address = Path(directory) / 'control.sock'
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(address))
            listener.listen(1)
            states = {'GPU-a': {'index': '0'}, 'GPU-b': {'index': '1'}}
            locks = {'GPU-a': Mock(), 'GPU-b': Mock()}
            result = []
            def serve():
                result.append(gpu.handle_release(listener, states, locks, 'GPU-a', 'GPU-a'))
            with patch.object(gpu, 'stop_worker') as stop, patch.object(gpu, 'control_path', return_value=address):
                thread = threading.Thread(target=serve)
                thread.start()
                try:
                    self.assertEqual(gpu.release_gpu(123, '0'), 0)
                finally:
                    thread.join(5)
                    listener.close()
                self.assertFalse(thread.is_alive())
                stop.assert_called_once()
            self.assertEqual(result, [(None, None)])
            self.assertEqual(list(states), ['GPU-b'])

    def controller(self, mode):
        processes = []
        pipes = []
        stopped = []
        class Pipe:
            def __init__(self):
                self.events = [('online', 100)]
                self.commands = []
            def poll(self): return bool(self.events)
            def recv(self): return self.events.pop(0)
            def send(self, command):
                self.commands.append(command)
                if command == 'claim':
                    self.events.append(('ready', {'pid': 100, 'end': None}))
            def close(self): pass
        def make_pipe():
            pipe = Pipe()
            pipes.append(pipe)
            return pipe, Mock()
        def make_process(**kwargs):
            process = Mock(pid=100)
            process.is_alive.return_value = True
            processes.append(process)
            return process
        context = Mock()
        context.Pipe.side_effect = make_pipe
        context.Process.side_effect = make_process
        rows = [dict(index=str(i), uuid=f'GPU-{i}', free=40, used=0, total=40, pids=set()) for i in range(2)]
        turns = 0
        def release(listener, states, locks, candidate, winner):
            nonlocal turns
            turns += 1
            if turns == 4:
                self.assertEqual(len(states), 1 if mode == 'any' else 2)
                self.assertTrue(all(s['status'] == 'computing' for s in states.values()))
                for u in list(states):
                    gpu.stop_worker(states.pop(u))
                    locks.pop(u).close()
            return candidate, winner
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(gpu.Path, 'home', return_value=Path(directory)), \
             patch.object(gpu, 'CLAIM_MODE', mode), \
             patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0,1'), \
             patch.object(sys, 'argv', ['llama']), \
             patch.object(gpu, 'snapshot', return_value=rows), \
             patch.object(gpu.mp, 'get_context', return_value=context), \
             patch.object(gpu, 'handle_release', side_effect=release), \
             patch.object(gpu, 'stop_worker', side_effect=lambda s: stopped.append(s['index'])), \
             patch.object(gpu.time, 'sleep'):
            self.assertEqual(gpu.main(), 0)
            self.assertFalse(gpu.control_path(os.getpid()).exists())
        self.assertCountEqual(stopped, ['0', '1'])
        self.assertEqual(pipes[0].commands, ['claim'])
        self.assertEqual(pipes[1].commands, [] if mode == 'any' else ['claim'])

    def test_any_claims_one_and_releases_other(self): self.controller('any')
    def test_all_keeps_both_computing(self): self.controller('all')

    def test_fixed_memory_allocates_exact_bytes_and_cleans(self):
        torch = Mock()
        torch.cuda.mem_get_info.return_value = (4 * gpu.GIB, 8 * gpu.GIB)
        with patch.dict(sys.modules, torch=torch), patch.object(gpu.threading, 'Event') as event:
            event.return_value.is_set.return_value = False
            self.assertEqual(gpu.hold_memory(.6, 'GPU-test'), 0)
        self.assertEqual(sum(c.args[0] for c in torch.empty.call_args_list), int(.6 * gpu.GIB))
        event.return_value.wait.assert_called_once()
        torch.mm.assert_not_called()
        torch.cuda.empty_cache.assert_called_once()

    def test_worker_grows_then_rolls_back_on_failure(self):
        torch = Mock()
        torch.cuda.mem_get_info.return_value = (20 * gpu.GIB, 40 * gpu.GIB)
        block = Mock()
        block.numel.return_value = 256 * 1024**2
        torch.empty.side_effect = [block, RuntimeError('simulated OOM')]
        pipe = Mock()
        pipe.poll.return_value = True
        pipe.recv.side_effect = ['grow', 'stop']
        with patch.dict(sys.modules, torch=torch), patch.dict(os.environ):
            gpu.worker('GPU-test', pipe)
        events = [c.args[0] for c in pipe.send.call_args_list]
        self.assertEqual([e[0] for e in events], ['online', 'retry'])
        self.assertIn('保留张量=0.00', events[-1][1])
        torch.cuda.empty_cache.assert_called()

    def test_worker_unlimited_compute_and_cleanup(self):
        cuda = Mock()
        cuda.mem_get_info.return_value = (2 * gpu.GIB, 4 * gpu.GIB)
        cuda.memory_allocated.return_value = 2 * gpu.GIB
        torch = Mock(cuda=cuda)
        pipe = Mock()
        pipe.poll.return_value = True
        pipe.recv.side_effect = ['claim', 'stop']
        with patch.dict(sys.modules, torch=torch), \
             patch.dict(os.environ), patch.object(gpu, 'snapshot', return_value=[{'uuid': 'GPU-test', 'pids': set()}]), \
             patch.object(gpu, 'COMPUTE_HOURS', None):
            gpu.worker('GPU-test', pipe)
        events = [call.args[0][0] for call in pipe.send.call_args_list]
        self.assertEqual(events, ['online', 'ready'])
        self.assertIsNone(pipe.send.call_args_list[-1].args[0][1]['end'])
        self.assertGreaterEqual(torch.mm.call_count, 5)
        cuda.empty_cache.assert_called()
        pipe.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
