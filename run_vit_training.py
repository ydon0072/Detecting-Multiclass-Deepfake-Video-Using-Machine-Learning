"""Run the requested CUDA ViT experiment with load controls and durable logs."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    root = Path(__file__).resolve().parent
    out = root / 'runs/vit'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists() or (out / 'last.pt').exists():
        raise RuntimeError('Existing ViT experiment; inspect before restarting')
    status = {'state': 'running', 'pid': os.getpid(), 'started': time.time(), 'epochs': 30,
              'gpu_duty': 0.6, 'gpu_memory_fraction': 0.6}
    cmd = [sys.executable, '-u', str(root / 'transformer_train.py'),
           '--data-root', str(root / 'data/FaceForensics++'), '--splits-dir', str(root / 'data/splits'),
           '--output', str(out), '--device', 'cuda', '--amp', '--epochs', '30', '--batch-size', '1',
           '--frames', '16', '--workers', '2', '--lr', '0.00003',
           '--gpu-duty', '0.6', '--gpu-memory-fraction', '0.6']
    (out / 'command.json').write_text(json.dumps(cmd, indent=2), encoding='utf-8')
    def save():
        (out / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        with (out / 'console.log').open('w', encoding='utf-8') as log:
            p = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = p.pid
            save()
            code = p.wait()
        status.update(state='complete' if code == 0 else 'failed', exit_code=code, finished=time.time())
    finally:
        save()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__ == '__main__':
    main()
