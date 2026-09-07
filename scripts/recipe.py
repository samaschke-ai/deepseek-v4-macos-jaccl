#!/usr/bin/env python3
"""Explicit foreground launch; --dry-run prints argv without touching a runtime."""
import argparse
import json
import os
import socket
import subprocess
from pathlib import Path

def guard_start(role, host):
    name = 'ggml-rpc-server' if role == 'worker' else 'llama-server'
    result = subprocess.run(['/usr/bin/pgrep', '-x', name], capture_output=True)
    if result.returncode != 1:
        raise RuntimeError('Refusing existing process (or failed process inspection): '+name)
    with socket.socket() as sock:
        sock.bind((host if role == 'worker' else '127.0.0.1', 50052 if role == 'worker' else 8241))

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('role', choices=['worker', 'coordinator'])
    p.add_argument('--runtime', type=Path, required=True, help='Patched build directory')
    p.add_argument('--models', type=Path)
    p.add_argument('--rpc-host', required=True, help='Worker private Thunderbolt IPv4 address')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    import ipaddress
    try:
        ip = ipaddress.IPv4Address(a.rpc_host)
        if not any(ip in ipaddress.IPv4Network(n) for n in ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16']):
            raise ValueError('RPC requires an RFC1918 private Thunderbolt IPv4 address')
    except ValueError as e:
        p.error(str(e))
    a.runtime = a.runtime.expanduser().resolve()
    if a.models is not None:
        a.models = a.models.expanduser().resolve()
    if a.role == 'worker':
        args = [str(a.runtime/'bin/ggml-rpc-server'), '--host', a.rpc_host, '--port', '50052', '--device', 'MTL0', '-c']
    else:
        if a.models is None:
            p.error('--models is required for coordinator')
        m = json.loads((ROOT/'model-manifest.json').read_text())
        args = [str(a.runtime/'bin/llama-server'), '--rpc', a.rpc_host+':50052', '--device', 'MTL0,RPC0', '--model', str(a.models/m['model']), '--mmproj', str(a.models/m['projector']), '--split-mode', 'layer', '--tensor-split', '1,1', '--host', '127.0.0.1', '--port', '8241', '--ctx-size', '1048576', '--parallel', '4', '--n-gpu-layers', '999', '--flash-attn', 'on', '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--load-mode', 'none', '--cache-ram', '0', '--slots', '--batch-size', '2048', '--ubatch-size', '1024', '--alias', 'deepseek-v4-flash', '--verbosity', '4', '--spec-type', 'none', '--prefill-budget', '1024']
    if a.dry_run:
        print(json.dumps(args))
        return
    guard_start(a.role, a.rpc_host)
    if a.role == 'coordinator':
        for entry in m['files']:
            if (a.models/entry['path']).stat().st_size != entry['bytes']:
                p.error('Model file size mismatch: '+entry['path'])
    os.execv(args[0], args)

if __name__ == '__main__':
    main()
