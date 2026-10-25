"""Explicit new experiment from the best checkpoint, retaining optimizer moments and tuning only the last four blocks."""
import ctypes
import json
from pathlib import Path
import os
import subprocess
import sys
import time
import torch
from utils.runtime import save_checkpoint

def prepare_checkpoint(source, output):
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != 'vit_b_16_mean':
        raise ValueError('Expected ViT checkpoint')
    groups = checkpoint['optimizer']['param_groups']
    if len(groups) != 2:
        raise ValueError('Expected separate backbone and classifier optimizer groups')
    baseline = {'source': str(source), 'epoch': checkpoint['epoch'] + 1,
                'val': checkpoint['history'][-1]['val'], 'previous_config': dict(checkpoint['config'])}
    groups[0]['lr'] = 2e-6
    groups[1]['lr'] = 2e-5
    for group in groups:
        group['weight_decay'] = 0.01
    config = checkpoint['config']
    config.update(backbone_lr=groups[0]['lr'], lr=groups[1]['lr'],
                  early_stopping_patience=3, output=str(output), warmup_epochs=0,
                  train_last_blocks=4, weight_decay=0.01)
    checkpoint['bad_epochs'] = 0
    # Preserve the already measured best score: worse epochs cannot replace it.
    save_checkpoint(checkpoint, output/'restart.pt')
    save_checkpoint(checkpoint, output/'best.pt')
    (output/'baseline.json').write_text(json.dumps(baseline, indent=2), encoding='utf-8')
    return baseline, config

def main():
    root = Path(__file__).resolve().parent
    output = root/'runs/vit_partial_refine'
    if output.exists():
        raise RuntimeError('Run directory already exists; inspect before restarting')
    output.mkdir(parents=True)
    status = {'state':'preparing', 'pid':os.getpid(), 'started':time.time()}
    def save():
        temp = output/'status.tmp'
        temp.write_text(json.dumps(status,indent=2),encoding='utf-8')
        os.replace(temp,output/'status.json')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        save()
        source = root/'runs/vit_recovery_extended/full/best.pt'
        baseline, config = prepare_checkpoint(source, output)
        target = baseline['epoch'] + 6
        cmd = [sys.executable,'-u',str(root/'transformer_train.py'),
               '--data-root',str(root/'data/FaceForensics++'), '--splits-dir',str(root/'data/splits'),
               '--resume',str(output/'restart.pt'),'--epochs',str(target),'--output',str(output),
               '--device','cuda','--gpu-duty','0.6','--gpu-memory-fraction','0.6']
        (output/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
        with (output/'console.log').open('w',encoding='utf-8') as log:
            process = subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            status.update(state='running',child_pid=process.pid,baseline_epoch=baseline['epoch'],
                          baseline_accuracy=baseline['val']['accuracy'],baseline_macro_f1=baseline['val']['macro_f1'],
                          backbone_lr=config['backbone_lr'],classifier_lr=config['lr'],
                          max_additional_epochs=6,patience=3,train_last_blocks=4,weight_decay=0.01)
            save()
            code = process.wait()
        if code:
            raise RuntimeError(f'Training exited {code}; inspect console.log')
        history = json.loads((output/'history.json').read_text())
        best = max(history,key=lambda row:row['val']['macro_f1'])
        status.update(state='complete',additional_epochs=len(history)-baseline['epoch'],
                      best_epoch=best['epoch'],best_accuracy=best['val']['accuracy'],
                      best_macro_f1=best['val']['macro_f1'])
    except Exception as exc:
        status.update(state='failed',error=str(exc))
        raise
    finally:
        save()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__': main()
