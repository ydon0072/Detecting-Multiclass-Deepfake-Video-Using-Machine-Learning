"""Gated pilot followed by full ViT training; never tune on the test set."""
import ctypes
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

def main():
    root = Path(__file__).resolve().parent
    out = root / 'runs/vit_adjusted'
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'status.json'
    if status_path.exists():
        raise RuntimeError('Adjusted experiment already exists')
    status = {'state':'running','stage':'pilot','pid':os.getpid(),'started':time.time()}
    def save():
        temp = out / 'status.tmp'
        temp.write_text(json.dumps(status,indent=2),encoding='utf-8')
        os.replace(temp,status_path)
    def run(splitdir, destination, epochs):
        destination.mkdir(exist_ok=True)
        cmd = [sys.executable,'-u',str(root/'transformer_train.py'), '--data-root',str(root/'data/FaceForensics++'),
               '--splits-dir',str(splitdir),'--output',str(destination),'--epochs',str(epochs),
               '--device','cuda','--batch-size','1','--frames','16','--workers','2',
               '--lr','0.0003','--backbone-lr','0.000001','--warmup-epochs','2',
               '--accumulation-steps','8','--gpu-duty','0.6','--gpu-memory-fraction','0.6']
        (destination/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
        with (destination/'console.log').open('w',encoding='utf-8') as log:
            p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            status.update(child_pid=p.pid,log=str(destination/'console.log'))
            save()
            if p.wait(): raise RuntimeError(f'Training failed; see {destination}')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        save()
        splits=out/'pilot_splits'
        splits.mkdir(exist_ok=True)
        rng=random.Random(42)
        for name,n in [('train',40),('val',20),('test',None)]:
            pairs=json.loads((root/f'data/splits/{name}.json').read_text())
            (splits/f'{name}.json').write_text(json.dumps(rng.sample(pairs,n) if n else pairs))
        run(splits,out/'pilot',5)
        history=json.loads((out/'pilot/history.json').read_text())
        best=max(history,key=lambda e:e['val']['macro_f1'])
        matrix=best['val']['confusion_matrix']
        predicted_classes=sum(sum(row[c] for row in matrix)>0 for c in range(5))
        status.update(pilot_accuracy=best['val']['accuracy'],pilot_f1=best['val']['macro_f1'],predicted_classes=predicted_classes)
        # Sanity gate only, not a claim of statistical significance or optimal tuning.
        if best['val']['accuracy'] <= 0.25 or best['val']['macro_f1'] <= 0.20 or predicted_classes < 3:
            status.update(state='needs_review',stage='pilot_finished',reason='Pilot did not pass learning sanity gate; full training not started')
        else:
            status['stage']='full_training'
            save()
            run(root/'data/splits',out/'full',30)
            status.update(state='complete',stage='training_complete')
    except Exception as exc:
        status.update(state='failed',error=str(exc))
        raise
    finally:
        save()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    main()
