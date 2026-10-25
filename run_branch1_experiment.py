"""Validate FF++, tune on a training/validation subset, train, then test once."""
import concurrent.futures
import ctypes
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import cv2
from dataset.ffpp import FFPPDataset, read_splits

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/FaceForensics++'
SPLITS = ROOT / 'data/splits'
RUN = ROOT / 'runs/branch1_experiment'
STATUS = RUN / 'status.json'

def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)

def decode(path):
    cap = cv2.VideoCapture(str(path))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    try:
        if not cap.isOpened() or expected <= 0:
            return {'path': str(path), 'error': 'Cannot open video or invalid frame count'}
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                return {'path': str(path), 'error': f'Empty frame at {count}'}
            count += 1
        return {'path': str(path), 'frames': count, 'expected': expected,
                'error': None if count == expected else 'Decoded frame count mismatch'}
    finally:
        cap.release()

def main():
    RUN.mkdir(parents=True, exist_ok=True)
    if STATUS.exists():
        raise RuntimeError('Experiment already exists. Inspect it before restarting.')
    status = {'state': 'running', 'stage': 'validation', 'pid': os.getpid(), 'started': time.time()}
    def update(**values):
        status.update(values, updated=time.time())
        write_json(STATUS, status)
    def command(script, args, folder):
        folder.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-u', str(ROOT / script), '--data-root', str(DATA), *args]
        write_json(folder / 'command.json', cmd)
        with (folder / 'console.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            update(child_pid=process.pid, log=str(folder / 'console.log'))
            code = process.wait()
        if code:
            raise RuntimeError(f'{script} exited {code}; see {folder / "console.log"}')
    # Keep Windows awake while work is active; the display may turn off.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        update()
        cv2.setNumThreads(1)
        splits = read_splits(SPLITS)
        datasets = {s: FFPPDataset(DATA, pairs) for s, pairs in splits.items()}
        paths = [p for ds in datasets.values() for p, _ in ds.samples]
        if len(paths) != 5000 or len(set(paths)) != 5000:
            raise RuntimeError('Expected exactly 5000 distinct official videos')
        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(decode, p) for p in paths]
            for future in concurrent.futures.as_completed(futures):
                records.append(future.result())
                if len(records) % 25 == 0:
                    update(validated=len(records), total=5000)
        failures = [r for r in records if r['error']]
        write_json(RUN / 'validation.json', {'counts': {s: ds.counts for s, ds in datasets.items()},
                                           'failures': failures, 'videos': records})
        if failures:
            raise RuntimeError(f'{len(failures)} videos failed decoding. Training was not started.')
        # Stratified-by-source subsets: all five classes inherit the same pairs.
        subset = RUN / 'tuning_splits'
        subset.mkdir(exist_ok=True)
        rng = random.Random(42)
        for split, pairs in splits.items():
            selected = rng.sample(pairs, 20 if split == 'train' else 10) if split != 'test' else pairs
            write_json(subset / f'{split}.json', selected)
        trials = []
        for lr in ['0.0001', '0.00003']:
            folder = RUN / f'tune_lr_{lr}'
            update(stage='tuning', learning_rate=lr)
            command('cnn_lstm_train.py', ['--splits-dir', str(subset), '--epochs', '3',
                    '--batch-size', '2', '--frames', '16', '--workers', '2', '--lr', lr,
                    '--output', str(folder)], folder)
            history = json.loads((folder / 'history.json').read_text())
            trials.append({'lr': lr, 'best_val_macro_f1': max(e['val']['macro_f1'] for e in history)})
            write_json(RUN / 'tuning_results.json', trials)
        best = max(trials, key=lambda t: t['best_val_macro_f1'])
        full = RUN / 'full'
        update(stage='full_training', selected_lr=best['lr'], epochs=30)
        # Fresh initialization: no continuation from a subset's checkpoint.
        command('cnn_lstm_train.py', ['--splits-dir', str(SPLITS), '--epochs', '30',
                '--batch-size', '2', '--frames', '16', '--workers', '2', '--lr', best['lr'],
                '--output', str(full)], full)
        update(stage='test')
        test = RUN / 'test'
        command('cnn_lstm_test.py', ['--splits-dir', str(SPLITS), '--checkpoint', str(full / 'best.pt'),
                '--workers', '2', '--output', str(test)], test)
        metrics = json.loads((test / 'test_metrics.json').read_text())
        update(state='complete', stage='complete', test_accuracy=metrics['accuracy'],
               test_macro_f1=metrics['macro_f1'], result=str(test / 'test_metrics.json'))
    except Exception as exc:
        update(state='failed', error=str(exc))
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__ == '__main__':
    main()
