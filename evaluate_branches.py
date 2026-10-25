"""Evaluate both independent branches on identical decoded FF++ test clips."""
import csv
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CLASSES
from dataset.ffpp import FFPPDataset, read_splits
from model.cnn_lstm import CNNLSTM
from model.vit import VideoViT
from utils.metrics import Metrics
from utils.runtime import seed_everything, seed_worker

def main():
    root=Path(__file__).resolve().parent
    seed_everything(42)
    device=torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(0.6)
    splits=read_splits(root/'data/splits')
    digest=hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest()
    specifications=[('CNN+LSTM',root/'runs/branch1_experiment/full/best.pt',root/'runs/branch1_experiment/test'),
                    ('ViT',root/'runs/vit_low_lr/best.pt',root/'runs/vit_low_lr/test')]
    runs=[]
    preprocessing=None
    for name,path,out in specifications:
        if (out/'test_metrics.json').exists(): raise RuntimeError(f'Existing evaluation: {out}')
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        config=checkpoint['config']
        assert checkpoint['classes']==CLASSES and checkpoint['split_hash']==digest
        settings=(config['compression'],config['frames'],config['image_size'])
        if preprocessing is not None and settings!=preprocessing: raise ValueError('Preprocessing differs')
        preprocessing=settings
        if name=='CNN+LSTM':
            model=CNNLSTM(config['hidden_size'],pretrained=False,freeze_backbone=config['freeze_backbone'])
        else:
            assert checkpoint['architecture']=='vit_b_16_mean'
            model=VideoViT(pretrained=False,freeze_backbone=config['freeze_backbone'])
        model.load_state_dict(checkpoint['model'],strict=True)
        model.to(device).eval()
        runs.append(dict(name=name,model=model,metrics=Metrics(),rows=[],path=path,out=out,epoch=checkpoint['epoch']+1))
        del checkpoint
    dataset=FFPPDataset(root/'data/FaceForensics++',splits['test'],*preprocessing,training=False)
    assert len(dataset)==700 and dataset.counts==[140]*5
    loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=4,pin_memory=True,
                      worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42))
    print('700 official test videos; two separate models; identical clips; CUDA AMP; duty=0.6',flush=True)
    offset=0
    with torch.inference_mode():
        for clips,labels in tqdm(loader,desc='Independent branches'):
            torch.cuda.synchronize();started=time.perf_counter()
            clips,labels=clips.to(device,non_blocking=True),labels.to(device,non_blocking=True)
            for run in runs:
                with torch.autocast('cuda'):
                    logits=run['model'](clips)
                    loss=torch.nn.functional.cross_entropy(logits,labels)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss): raise RuntimeError('Nonfinite outputs')
                run['metrics'].update(logits,labels,loss.item())
                for i,probability in enumerate(logits.float().softmax(1).cpu().tolist()):
                    path,truth=dataset.samples[offset+i]
                    pred=max(range(5),key=probability.__getitem__)
                    run['rows'].append([str(path.relative_to(root/'data/FaceForensics++')),truth,CLASSES[truth],pred,CLASSES[pred],*probability])
            offset+=len(labels)
            torch.cuda.synchronize()
            time.sleep((time.perf_counter()-started)*(1/0.6-1))
    for run in runs:
        result=run['metrics'].compute()
        matrix=torch.tensor(result['confusion_matrix'],dtype=torch.float64)
        precision=matrix.diag()/matrix.sum(0).clamp_min(1)
        recall=matrix.diag()/matrix.sum(1).clamp_min(1)
        result.update(macro_precision=precision.mean().item(),macro_recall=recall.mean().item(),
                      per_class={name:dict(precision=precision[i].item(),recall=recall[i].item(),f1=result['per_class_f1'][i],support=140) for i,name in enumerate(CLASSES)},
                      classes=CLASSES,split='test',split_hash=digest,checkpoint=str(run['path']),checkpoint_epoch=run['epoch'],
                      compression=preprocessing[0],frames=preprocessing[1],image_size=preprocessing[2],
                      class_counts=dict(zip(CLASSES,dataset.counts)),incomplete_test_set=False,device='cuda',amp=True)
        run['out'].mkdir(parents=True,exist_ok=True)
        (run['out']/'test_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        with (run['out']/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f)
            writer.writerow(['video','true_id','true_class','predicted_id','predicted_class',*[f'prob_{name}' for name in CLASSES]])
            writer.writerows(run['rows'])
        with (run['out']/'confusion_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(['true / predicted',*CLASSES])
            writer.writerows([name,*row] for name,row in zip(CLASSES,result['confusion_matrix']))
        print(run['name'],json.dumps(result),flush=True)

if __name__=='__main__': main()
