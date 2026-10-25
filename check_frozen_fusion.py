"""Read-only checkpoint compatibility and single-step frozen-branch smoke test."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from dataset.ffpp import FFPPDataset, read_splits
from utils.fusion_setup import build_model, initialize_branches
from transformer_train import run_epoch

def main():
    root=Path(r'E:\study\论文代码')
    config=dict(hidden_size=256,fusion_dim=256,attention_heads=4,mlp_hidden=128,mlp_dropout=0.3,
                freeze_branches=True,frames=16,image_size=224,compression='c23',pretrained=False,
                cnn_checkpoint=str(root/'runs/branch1_experiment/full/best.pt'),
                vit_checkpoint=str(root/'runs/vit_low_lr/best.pt'))
    splits=read_splits(root/'data/splits')
    split_hash=hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest()
    model=build_model(config)
    print(initialize_branches(model,SimpleNamespace(**config),split_hash))
    torch.cuda.set_per_process_memory_fraction(0.6)
    model=model.cuda()
    ds=FFPPDataset(root/'data/FaceForensics++',splits['train'][:1],frames=16)
    x,y=ds[0]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    before=model.classifier[-1].weight.detach().clone()
    # Two copies exercise the actual batch-size=2, 16-frame allocation.
    metrics=run_epoch(model,[(torch.stack([x,x]),torch.tensor([y,y]))],torch.device('cuda'),True,
                      optimizer,scaler,gpu_duty=0.6)
    assert metrics['optimizer_updates']==1
    assert not torch.equal(before,model.classifier[-1].weight)
    assert not model.cnn_lstm.training and not model.vit.training
    assert all(not p.requires_grad and p.grad is None for p in model.cnn_lstm.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.vit.parameters())
    print('PASS: original checkpoints compatible; both branches frozen; MLP updates; GPU memory fits')
    print('Peak reserved GiB',torch.cuda.max_memory_reserved()/2**30)

if __name__=='__main__': main()
