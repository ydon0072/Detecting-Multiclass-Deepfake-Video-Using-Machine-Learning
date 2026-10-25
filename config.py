"""Shared configuration and stable class order."""
import argparse

CLASSES = ['Real', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

def parse_args():
    p = argparse.ArgumentParser(description='FF++ ResNet18 + LSTM five-class training')
    p.add_argument('--data-root', required=True)
    p.add_argument('--splits-dir', required=True, help='Official train.json, val.json, test.json')
    p.add_argument('--compression', choices=['raw', 'c23', 'c40'], default='c23')
    p.add_argument('--output', default='runs/cnn_lstm')
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
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--freeze-backbone', action='store_true')
    p.add_argument('--resume', help='Trusted local checkpoint only')
    p.add_argument('--evaluate', choices=['val', 'test'])
    p.add_argument('--allow-missing', action='store_true', help='Explicitly permit incomplete dataset')
    a = p.parse_args()
    for name in ['epochs', 'batch_size', 'frames', 'hidden_size']:
        if getattr(a, name) < 1:
            p.error(f'{name} must be positive')
    if a.image_size < 32 or a.workers < 0 or a.lr <= 0 or a.weight_decay < 0:
        p.error('Invalid image size, workers, learning rate or weight decay')
    if a.evaluate and not a.resume:
        p.error('--evaluate requires --resume')
    return a
