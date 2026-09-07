import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

class RecipeTests(unittest.TestCase):
    def test_coordinator_dry_run_has_proven_flags(self):
        import subprocess, sys, json
        r = subprocess.run([sys.executable, str(ROOT/'scripts/recipe.py'), 'coordinator', '--runtime', '/tmp/runtime', '--models', '/tmp/models', '--rpc-host', '192.168.0.2', '--dry-run'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        args = json.loads(r.stdout)
        for flag, value in {'--split-mode':'layer','--tensor-split':'1,1','--prefill-budget':'1024','--parallel':'4','--ctx-size':'1048576','--cache-type-k':'f16','--cache-type-v':'f16','--spec-type':'none','--host':'127.0.0.1','--load-mode':'none','--cache-ram':'0','--device':'MTL0,RPC0'}.items():
            self.assertEqual(args[args.index(flag)+1], value)

    def test_public_rpc_rejected(self):
        import subprocess, sys
        for host in ['0.0.0.0', '8.8.8.8', '127.0.0.1', 'worker.example.invalid']:
            r = subprocess.run([sys.executable, str(ROOT/'scripts/recipe.py'), 'worker', '--runtime', '/tmp/runtime', '--rpc-host', host, '--dry-run'], capture_output=True)
            self.assertNotEqual(r.returncode, 0, host)

    def test_manifest_validation_rejects_corruption(self):
        spec = importlib.util.spec_from_file_location('preflight', ROOT/'scripts/preflight.py')
        self.assertTrue((ROOT/'scripts/preflight.py').exists(), 'preflight missing')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp)/'a.gguf'; p.write_bytes(b'GGUF')
            manifest = {'files':[{'path':'a.gguf','bytes':4,'sha256':hashlib.sha256(b'GGUF').hexdigest()}]}
            module.verify_files(pathlib.Path(tmp), manifest)
            p.write_bytes(b'BAD!')
            with self.assertRaises(ValueError): module.verify_files(pathlib.Path(tmp), manifest)
            p.unlink()
            with self.assertRaises((ValueError, FileNotFoundError)): module.verify_files(pathlib.Path(tmp), manifest)

    def test_start_guard_refuses_existing_process(self):
        spec = importlib.util.spec_from_file_location('recipe', ROOT/'scripts/recipe.py')
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'guard_start'))
        from unittest.mock import patch
        import subprocess
        with patch.object(module.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaises(RuntimeError): module.guard_start('worker', '192.168.0.2')

    def test_semantic_verifier_requires_explicit_inference(self):
        import subprocess, sys
        r = subprocess.run([sys.executable, str(ROOT/'scripts/verify.py'), '--url', 'http://127.0.0.1:1'], capture_output=True, text=True)
        self.assertIn('--inference', r.stderr)

    def test_results_checker_runs(self):
        import subprocess, sys
        r = subprocess.run([sys.executable, str(ROOT/'scripts/check_results.py')], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('72', r.stdout)

    def test_slot_capacity_is_enforced(self):
        spec = importlib.util.spec_from_file_location('preflight', ROOT/'scripts/preflight.py')
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'verify_slots'))
        good = [{'n_ctx':262144, 'is_processing':False} for _ in range(4)]
        module.verify_slots(good)
        good[0]['n_ctx'] = 1
        with self.assertRaises(ValueError): module.verify_slots(good)

if __name__ == '__main__': unittest.main()
