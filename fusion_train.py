import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fusion_config import CLASSES, parse_args
from dataset.ffpp import FFPPDataset, read_splits
from utils.fusion_setup import ARCHITECTURE, build_model, initialize_branches
from utils.metrics import Metrics
from utils.runtime import seed_everything, seed_worker, rng_state, restore_rng, save_checkpoint

from transformer_train import run_epoch

def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install a matching CUDA PyTorch build or explicitly use --device cpu.')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    amp = args.amp and device.type == 'cuda'
    seed_everything(args.seed)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if checkpoint:
        if checkpoint.get('architecture') != ARCHITECTURE:
            raise ValueError('Expected an attention fusion checkpoint')
        if checkpoint['classes'] != CLASSES:
            raise ValueError('Checkpoint class mapping differs')
        # Resume model/preprocessing/optimizer settings from its saved configuration.
        keys = ['hidden_size', 'fusion_dim', 'attention_heads', 'mlp_hidden', 'mlp_dropout', 'freeze_branches', 'frames', 'image_size', 'compression']
        if not args.evaluate:
            keys += ['batch_size', 'workers', 'seed', 'lr', 'weight_decay', 'amp', 'allow_missing']
        for key in keys:
            setattr(args, key, checkpoint['config'][key])
        if not args.evaluate:
            for key, default in [('accumulation_steps', 1), ('early_stopping_patience', 0)]:
                setattr(args, key, checkpoint['config'].get(key, default))
        amp = args.amp and device.type == 'cuda'
    splits = read_splits(args.splits_dir)
    split_hash = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    if checkpoint and split_hash != checkpoint['split_hash']:
        raise ValueError('Official split files differ from checkpoint')
    generator = torch.Generator().manual_seed(args.seed)
    loaders, manifests = {}, {}
    for split in ([args.evaluate] if args.evaluate else ['train', 'val']):
        ds = FFPPDataset(args.data_root, splits[split], args.compression, args.frames,
                         args.image_size, split == 'train', args.allow_missing)
        print(f'{split}: {dict(zip(CLASSES, ds.counts))}')
        manifests[split] = [(str(p.relative_to(Path(args.data_root))), y) for p, y in ds.samples]
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=split == 'train',
                                    num_workers=args.workers, pin_memory=device.type == 'cuda',
                                    worker_init_fn=seed_worker, generator=generator)
    if checkpoint and not args.evaluate and manifests != checkpoint['manifests']:
        raise ValueError('Dataset file selection changed since checkpoint')
    pretrained = args.pretrained and checkpoint is None and not (args.cnn_checkpoint and args.vit_checkpoint)
    model = build_model(vars(args), pretrained=pretrained)
    provenance = checkpoint.get('branch_initialization', {}) if checkpoint else initialize_branches(model, args, split_hash)
    model = model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=1024.0)
    start, best, history = 0, -1.0, []
    bad_epochs = 0
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        if not args.evaluate:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            start, best, history = checkpoint['epoch'] + 1, checkpoint['best_macro_f1'], checkpoint['history']
            bad_epochs = checkpoint.get('bad_epochs', 0)
            generator.set_state(checkpoint['loader_rng'])
            restore_rng(checkpoint['rng'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f'Device: {torch.cuda.get_device_name() if device.type == "cuda" else "CPU"}; AMP: {amp}')
    if args.evaluate:
        result = run_epoch(model, loaders[args.evaluate], device, amp, gpu_duty=args.gpu_duty)
        result.update(classes=CLASSES, checkpoint=str(Path(args.resume).resolve()), split_hash=split_hash)
        (output / f'{args.evaluate}_metrics.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    if start >= args.epochs:
        raise ValueError(f'Checkpoint already completed {start} epochs; --epochs is the total target')
    if not args.resume and (output / 'last.pt').exists():
        raise FileExistsError('Output already contains a run; use --resume or a different --output')
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    (output / 'manifest.json').write_text(json.dumps(manifests, indent=2))
    for epoch in range(start, args.epochs):
        train = run_epoch(model, loaders['train'], device, amp, optimizer, scaler, args.gpu_duty, args.accumulation_steps)
        val = run_epoch(model, loaders['val'], device, amp, gpu_duty=args.gpu_duty)
        improved = val['macro_f1'] > best
        bad_epochs = 0 if improved else bad_epochs + 1
        best = max(best, val['macro_f1'])
        history.append({'epoch': epoch + 1, 'train': train, 'val': val})
        payload = {'architecture': ARCHITECTURE, 'branch_initialization': provenance, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scaler': scaler.state_dict(), 'best_macro_f1': best, 'config': vars(args),
                   'classes': CLASSES, 'split_hash': split_hash, 'manifests': manifests,
                   'rng': rng_state(), 'loader_rng': generator.get_state(), 'history': history, 'bad_epochs': bad_epochs}
        save_checkpoint(payload, output / 'last.pt')
        if improved:
            save_checkpoint(payload, output / 'best.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'Epoch {epoch+1}/{args.epochs}: train acc={train["accuracy"]:.4f} F1={train["macro_f1"]:.4f}; val acc={val["accuracy"]:.4f} F1={val["macro_f1"]:.4f}')
        if args.early_stopping_patience and bad_epochs >= args.early_stopping_patience:
            print(f'Early stopping after {bad_epochs} epochs without validation macro F1 improvement.')
            break

if __name__ == '__main__':
    main()


