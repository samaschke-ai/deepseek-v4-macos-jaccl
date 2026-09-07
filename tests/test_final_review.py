"""Offline review regressions: no serving runtime or RPC connections."""
import base64
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/(name+'.py'))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FinalReviewTests(unittest.TestCase):
    def test_results_reject_duplicate_summary_coverage(self):
        module = load('check_results')
        data = json.loads((ROOT/'results/vision-quant-comparison.json').read_text())
        data['summary'][-1] = data['summary'][0].copy()
        with patch.object(module.json, 'loads', return_value=data):
            with self.assertRaises(AssertionError):
                module.main()

    def run_fake_http(self, bad_marker=False):
        module = load('verify')
        requests = []
        def fake_urlopen(request, timeout):
            self.assertEqual(request.full_url, 'http://offline.invalid/v1/chat/completions')
            payload = json.loads(request.data)
            self.assertFalse(payload['stream'])
            self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
            content = payload['messages'][0]['content']
            requests.append(content)
            if isinstance(content, list):
                image = content[1]['image_url']['url']
                self.assertTrue(base64.b64decode(image.split(',', 1)[1]).startswith(b'\x89PNG\r\n\x1a\n'))
                answer = 'left red, right blue'
            else:
                answer = content.removeprefix('Reply with exactly ').rstrip('.')
                if bad_marker and answer == 'DELTA':
                    answer = 'ALPHA'
            return io.BytesIO(json.dumps({'model':'deepseek-v4-flash', 'choices':[{'message':{'content':'</think>'+answer}}]}).encode())
        output = io.StringIO()
        with patch.object(module, 'urlopen', side_effect=fake_urlopen), patch.object(sys, 'argv', ['verify.py', '--url', 'http://offline.invalid', '--inference']), contextlib.redirect_stdout(output):
            if bad_marker:
                with self.assertRaisesRegex(ValueError, 'Concurrent marker check failed'):
                    module.main()
            else:
                module.main()
                self.assertIn('PASS: text, real image, four concurrent markers', output.getvalue())
        self.assertEqual(len(requests), 6)

    def test_fake_http_text_vision_markers(self):
        self.run_fake_http()

    def test_fake_http_rejects_crossed_marker(self):
        self.run_fake_http(bad_marker=True)

    def test_guard_checks_bind_without_connecting(self):
        module = load('recipe')
        sock = MagicMock()
        with patch.object(module.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)), patch.object(module.socket, 'socket', return_value=sock):
            module.guard_start('worker', '192.168.0.2')
            sock.__enter__.return_value.bind.assert_called_once_with(('192.168.0.2', 50052))
            sock.__enter__.return_value.connect.assert_not_called()
            sock.__enter__.return_value.bind.side_effect = OSError('occupied')
            with self.assertRaises(OSError):
                module.guard_start('coordinator', '192.168.0.2')

    def test_slots_reject_busy_or_wrong_count(self):
        module = load('preflight')
        good = [{'n_ctx':262144, 'is_processing':False} for _ in range(4)]
        module.verify_slots(good)
        with self.assertRaises(ValueError):
            module.verify_slots(good[:3])
        good[2]['is_processing'] = True
        with self.assertRaises(ValueError):
            module.verify_slots(good)


if __name__ == '__main__':
    unittest.main()
