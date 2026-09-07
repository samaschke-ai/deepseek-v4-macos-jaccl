import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

class LaunchIntegrityTests(unittest.TestCase):
    def test_launch_rejects_same_size_corruption(self):
        spec = importlib.util.spec_from_file_location('recipe_integrity', ROOT/'scripts/recipe.py')
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'a.gguf').write_bytes(b'BAD!')
            manifest = {'model':'a.gguf', 'projector':'a.gguf', 'files':[{'path':'a.gguf','bytes':4,'sha256':hashlib.sha256(b'GGUF').hexdigest()}]}
            (root/'model-manifest.json').write_text(json.dumps(manifest))
            argv = ['recipe.py','coordinator','--runtime',tmp,'--models',tmp,'--rpc-host','192.168.0.2']
            with patch.object(module,'ROOT',root), patch.object(module,'guard_start'), patch.object(module.os,'execv') as execute, patch.object(sys,'argv',argv):
                with self.assertRaises((ValueError, SystemExit)):
                    module.main()
                execute.assert_not_called()
                (root/'a.gguf').write_bytes(b'GGUF')
                module.main()
                execute.assert_called_once()
