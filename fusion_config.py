"""Shared configuration and stable class order."""
import argparse

CLASSES = ['Real', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

def parse_args():
    p = argparse.ArgumentParser(description='FF++ CNN+LSTM and ViT attention fusion')
    p.add_argument('--data-root', required=True)
    p.add_argument('--splits-dir', required=True, help='Official train.json, val.json, test.json')
    p.add_argument('--compression', choices=['raw', 'c23', 'c40'], default='c23')
    p.add_argument('--output', default='runs/fusion')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--frames', type=int, default=16)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--hidden-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=0, help='Windows: start at 0, then try 2 or 4')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--gpu-duty', type=float, default=1.0)
    p.add_argument('--gpu-memory-fraction', type=float, default=1.0)
    p.add_argument('--accumulation-steps', type=int, default=1)
    p.add_argument('--early-stopping-patience', type=int, default=0)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--freeze-branches', action='store_true', help='Train only projection, attention and fusion classifier')
    p.add_argument('--cnn-checkpoint', help='Trusted standalone CNN+LSTM checkpoint')
    p.add_argument('--vit-checkpoint', help='Trusted standalone ViT checkpoint')
    p.add_argument('--fusion-dim', type=int, default=256)
    p.add_argument('--attention-heads', type=int, default=4)
    p.add_argument('--mlp-hidden', type=int, default=128)
    p.add_argument('--mlp-dropout', type=float, default=0.3)
    p.add_argument('--resume', help='Trusted local checkpoint only')
    p.add_argument('--evaluate', choices=['val', 'test'])
    p.add_argument('--allow-missing', action='store_true', help='Explicitly permit incomplete dataset')
    a = p.parse_args()
    if not 0 < a.gpu_duty <= 1 or not 0 < a.gpu_memory_fraction <= 1:
        p.error('GPU fractions must be in (0,1]')
    if a.accumulation_steps < 1 or a.early_stopping_patience < 0:
        p.error('Invalid accumulation or early stopping setting')
    for name in ['epochs', 'batch_size', 'frames', 'hidden_size']:
        if getattr(a, name) < 1:
            p.error(f'{name} must be positive')
    if a.image_size != 224 or a.workers < 0 or a.lr <= 0 or a.weight_decay < 0:
        p.error('Invalid image size, workers, learning rate or weight decay')
    if a.evaluate and not a.resume:
        p.error('--evaluate requires --resume')
    if a.fusion_dim < 1 or a.attention_heads < 1 or a.fusion_dim % a.attention_heads:
        p.error('fusion-dim must be positive and divisible by attention-heads')
    if a.mlp_hidden < 1 or not 0 <= a.mlp_dropout < 1:
        p.error('mlp-hidden must be positive and mlp-dropout must be in [0,1)')
    if a.resume and (a.cnn_checkpoint or a.vit_checkpoint):
        p.error('Resume restores the full fusion model; do not supply branch checkpoints')
    if a.freeze_branches and not a.resume and not (a.cnn_checkpoint and a.vit_checkpoint):
        p.error('Freezing branches requires both trained branch checkpoints')
    return a

