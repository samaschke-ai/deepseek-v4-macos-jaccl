#!/usr/bin/env python3
"""Read-only file integrity and endpoint inspection. Never generates tokens."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

def verify_files(root, manifest):
    for entry in manifest['files']:
        p = root/entry['path']
        if p.stat().st_size != entry['bytes']:
            raise ValueError('Size mismatch: '+str(p))
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                h.update(block)
        if h.hexdigest() != entry['sha256']:
            raise ValueError('SHA256 mismatch: '+str(p))
    return len(manifest['files'])

def verify_slots(value):
    if len(value) != 4 or any(x.get('is_processing') is not False or x.get('n_ctx') != 262144 for x in value):
        raise ValueError('Expected four idle slots with n_ctx=262144; do not replace a busy server')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=Path)
    p.add_argument('--url', help='Read health, model identity and slots; loopback URL recommended')
    a = p.parse_args()
    if not a.models and not a.url:
        p.error('supply --models and/or --url')
    if a.models:
        print(json.dumps({'verified_files':verify_files(a.models, json.loads((ROOT/'model-manifest.json').read_text()))}))
    if a.url:
        for endpoint in ['/health', '/v1/models', '/slots']:
            with urlopen(a.url.rstrip('/')+endpoint, timeout=10) as response:
                value = json.load(response)
            print(json.dumps({endpoint:value}))
            if endpoint == '/health' and value.get('status') != 'ok':
                raise ValueError('Not healthy')
            if endpoint == '/v1/models' and 'deepseek-v4-flash' not in [x['id'] for x in value['data']]:
                raise ValueError('Wrong model alias')
            if endpoint == '/slots':
                verify_slots(value)

if __name__ == '__main__': main()
