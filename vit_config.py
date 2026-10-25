"""Shared configuration and stable class order."""
import argparse

from config import CLASSES

def parse_args():
    p = argparse.ArgumentParser(description='FF++ ViT-B/16 video five-class training')
    p.add_argument('--data-root', required=True)
    p.add_argument('--splits-dir', required=True, help='Official train.json, val.json, test.json')
    p.add_argument('--compression', choices=['raw', 'c23', 'c40'], default='c23')
    p.add_argument('--output', default='runs/vit')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--frames', type=int, default=16)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, default=None)
    p.add_argument('--warmup-epochs', type=int, default=0)
    p.add_argument('--accumulation-steps', type=int, default=1)
    p.add_argument('--train-last-blocks', type=int, default=12, choices=range(1,13))
    p.add_argument('--early-stopping-patience', type=int, default=0, help='0 disables early stopping')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=0, help='Windows: start at 0, then try 2 or 4')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--gpu-duty', type=float, default=1.0, help='Approximate compute duty cycle, not an exact utilization cap')
    p.add_argument('--gpu-memory-fraction', type=float, default=1.0, help='PyTorch allocator limit, excluding other apps')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--freeze-backbone', action='store_true')
    p.add_argument('--resume', help='Trusted local checkpoint only')
    p.add_argument('--evaluate', choices=['val', 'test'])
    p.add_argument('--allow-missing', action='store_true', help='Explicitly permit incomplete dataset')
    a = p.parse_args()
    if a.early_stopping_patience < 0:
        p.error('early-stopping-patience must be nonnegative')
    if a.accumulation_steps < 1 or a.warmup_epochs < 0 or (a.backbone_lr is not None and a.backbone_lr <= 0):
        p.error('Invalid accumulation, warmup or backbone learning rate')
    if not 0 < a.gpu_duty <= 1 or not 0 < a.gpu_memory_fraction <= 1:
        p.error('GPU duty and memory fraction must be in (0,1]')
    for name in ['epochs', 'batch_size', 'frames']:
        if getattr(a, name) < 1:
            p.error(f'{name} must be positive')
    if a.image_size != 224 or a.workers < 0 or a.lr <= 0 or a.weight_decay < 0:
        p.error('Invalid image size, workers, learning rate or weight decay')
    if a.evaluate and not a.resume:
        p.error('--evaluate requires --resume')
    return a

