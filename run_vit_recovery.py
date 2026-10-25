"""Training-only memorization check, then validation-only model selection."""
import ctypes
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from dataset.ffpp import FFPPDataset, read_splits
from model.vit import VideoViT
from transformer_train import run_epoch
from utils.runtime import seed_everything

def main():
    root=Path(__file__).resolve().parent
    out=root/'runs/vit_recovery'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists(): raise RuntimeError('Run already exists')
    status={'state':'running','stage':'memorization','pid':os.getpid()}
    def save():
        status['updated']=time.time()
        temp=out/'status.tmp'
        temp.write_text(json.dumps(status,indent=2))
        os.replace(temp,out/'status.json')
    def train(splitdir,folder,blocks,lr,epochs):
        folder.mkdir(exist_ok=True)
        cmd=[sys.executable,'-u',str(root/'transformer_train.py'),'--data-root',str(root/'data/FaceForensics++'),
             '--splits-dir',str(splitdir),'--output',str(folder),'--epochs',str(epochs),
             '--frames','16','--batch-size','1','--workers','2','--accumulation-steps','8',
             '--warmup-epochs','1','--backbone-lr',str(lr),'--lr','0.0001','--train-last-blocks',str(blocks),
             '--gpu-duty','0.6','--gpu-memory-fraction','0.6']
        (folder/'command.json').write_text(json.dumps(cmd,indent=2))
        with (folder/'console.log').open('w') as f:
            p=subprocess.Popen(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
            status.update(child_pid=p.pid,log=str(folder/'console.log'));save()
            if p.wait(): raise RuntimeError(f'Training failed: {folder}')
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        save();seed_everything(42)
        torch.cuda.set_per_process_memory_fraction(0.6)
        splits=read_splits(root/'data/splits')
        # Ten clips from one TRAIN source pair, all five classes, no augmentation.
        ds=FFPPDataset(root/'data/FaceForensics++',splits['train'][:1],frames=16,training=False)
        clips,labels=zip(*(ds[i] for i in range(len(ds))))
        cached=TensorDataset(torch.stack(clips),torch.tensor(labels))
        loader=DataLoader(cached,batch_size=1,shuffle=True)
        evaluation=DataLoader(cached,batch_size=1)
        model=VideoViT(pretrained=True).cuda()
        model.classifier[0].p=0.0
        model.backbone.requires_grad_(False)
        for block in list(model.backbone.encoder.layers)[-4:]: block.requires_grad_(True)
        model.backbone.encoder.ln.requires_grad_(True)
        optimizer=torch.optim.AdamW([{'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':1e-5},
                                    {'params':model.classifier.parameters(),'lr':1e-3}],weight_decay=0)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024)
        diagnostics=[]
        for epoch in range(30):
            run_epoch(model,loader,torch.device('cuda'),True,optimizer,scaler,0.6,2)
            measured=run_epoch(model,evaluation,torch.device('cuda'),True,gpu_duty=0.6)
            diagnostics.append(measured)
            (out/'memorization.json').write_text(json.dumps(diagnostics,indent=2))
            status.update(diagnostic_epoch=epoch+1,training_only_accuracy=measured['accuracy']);save()
            if measured['accuracy']>=0.9: break
        if measured['accuracy']<0.9:
            status.update(state='needs_review',reason='Cannot fit ten training clips; do not run long training');return
        del model,optimizer,scaler,clips,cached,loader,evaluation
        import gc
        gc.collect();torch.cuda.empty_cache()
        subset=out/'splits';subset.mkdir()
        rng=random.Random(42)
        for s,n in [('train',40),('val',20),('test',None)]:
            (subset/f'{s}.json').write_text(json.dumps(rng.sample(splits[s],n) if n else splits[s]))
        trials=[]
        for blocks,lr in [(4,1e-5),(12,5e-6)]:
            status.update(stage='validation_trials',blocks=blocks,backbone_lr=lr);save()
            folder=out/f'trial_blocks{blocks}'
            train(subset,folder,blocks,lr,12)
            history=json.loads((folder/'history.json').read_text())
            best=max(history,key=lambda x:x['val']['macro_f1'])
            trials.append({'blocks':blocks,'lr':lr,'epoch':best['epoch'],'val':best['val']})
            (out/'trials.json').write_text(json.dumps(trials,indent=2))
        best=max(trials,key=lambda x:x['val']['macro_f1'])
        if best['val']['accuracy']<=0.25 or best['val']['macro_f1']<=0.22:
            status.update(state='needs_review',reason='Validation still weak; preserved trials for diagnosis');return
        status.update(stage='full_training',selected=best);save()
        train(root/'data/splits',out/'full',best['blocks'],best['lr'],30)
        status.update(state='complete',stage='training_complete')
    except Exception as exc:
        status.update(state='failed',error=str(exc));raise
    finally:
        save();ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__': main()
