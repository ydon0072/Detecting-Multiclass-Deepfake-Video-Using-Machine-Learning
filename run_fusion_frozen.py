"""Frozen branches, trainable attention and MLP; validation-only selection."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    root=Path(__file__).resolve().parent
    out=root/'runs/fusion_frozen'
    if out.exists(): raise RuntimeError('Experiment already exists; inspect before restarting')
    out.mkdir(parents=True)
    status={'state':'starting','pid':os.getpid(),'started':time.time(),
            'epochs':30,'patience':5,'freeze_branches':True}
    def save():
        tmp=out/'status.tmp'
        tmp.write_text(json.dumps(status,indent=2),encoding='utf-8')
        os.replace(tmp,out/'status.json')
    cmd=[sys.executable,'-u',str(root/'fusion_train.py'),
         '--data-root',str(root/'data/FaceForensics++'),'--splits-dir',str(root/'data/splits'),
         '--cnn-checkpoint',str(root/'runs/branch1_experiment/full/best.pt'),
         '--vit-checkpoint',str(root/'runs/vit_low_lr/best.pt'),
         '--freeze-branches','--no-pretrained','--output',str(out),
         '--epochs','30','--early-stopping-patience','5',
         '--frames','16','--batch-size','2','--accumulation-steps','4',
         '--workers','2','--lr','0.0001','--weight-decay','0.0001',
         '--gpu-duty','0.6','--gpu-memory-fraction','0.6','--device','cuda','--amp']
    (out/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        save()
        with (out/'console.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            status.update(state='running',child_pid=process.pid);save()
            code=process.wait()
        if code: raise RuntimeError(f'Training exited {code}; see console.log')
        history=json.loads((out/'history.json').read_text())
        best=max(history,key=lambda e:e['val']['macro_f1'])
        status.update(state='complete',completed_epochs=len(history),best_epoch=best['epoch'],
                      best_val_accuracy=best['val']['accuracy'],best_val_macro_f1=best['val']['macro_f1'])
    except Exception as exc:
        status.update(state='failed',error=str(exc));raise
    finally:
        save();ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__': main()
