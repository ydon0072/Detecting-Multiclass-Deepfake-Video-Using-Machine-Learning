import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CLASSES, parse_args
from dataset.ffpp import FFPPDataset, read_splits
from model.cnn_lstm import CNNLSTM
from utils.metrics import Metrics
from utils.runtime import seed_everything, seed_worker, rng_state, restore_rng, save_checkpoint

def run_epoch(model, loader, device, amp, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    metrics = Metrics()
    with torch.set_grad_enabled(training):
        for videos, labels in tqdm(loader, desc='train' if training else 'eval'):
            videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(videos)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; check input data and learning rate')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            metrics.update(logits, labels, loss.detach().item())
    return metrics.compute()

def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install a matching CUDA PyTorch build or explicitly use --device cpu.')
    device = torch.device(args.device)
    amp = args.amp and device.type == 'cuda'
    seed_everything(args.seed)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if checkpoint:
        if checkpoint['classes'] != CLASSES:
            raise ValueError('Checkpoint class mapping differs')
        # Resume model/preprocessing/optimizer settings from its saved configuration.
        keys = ['hidden_size', 'freeze_backbone', 'frames', 'image_size', 'compression']
        if not args.evaluate:
            keys += ['batch_size', 'workers', 'seed', 'lr', 'weight_decay', 'amp', 'allow_missing']
        for key in keys:
            setattr(args, key, checkpoint['config'][key])
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
    model = CNNLSTM(args.hidden_size, args.pretrained and checkpoint is None, args.freeze_backbone).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=1024.0)
    start, best, history = 0, -1.0, []
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        if not args.evaluate:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            start, best, history = checkpoint['epoch'] + 1, checkpoint['best_macro_f1'], checkpoint['history']
            generator.set_state(checkpoint['loader_rng'])
            restore_rng(checkpoint['rng'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f'Device: {torch.cuda.get_device_name() if device.type == "cuda" else "CPU"}; AMP: {amp}')
    if args.evaluate:
        result = run_epoch(model, loaders[args.evaluate], device, amp)
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
        train = run_epoch(model, loaders['train'], device, amp, optimizer, scaler)
        val = run_epoch(model, loaders['val'], device, amp)
        improved = val['macro_f1'] > best
        best = max(best, val['macro_f1'])
        history.append({'epoch': epoch + 1, 'train': train, 'val': val})
        payload = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scaler': scaler.state_dict(), 'best_macro_f1': best, 'config': vars(args),
                   'classes': CLASSES, 'split_hash': split_hash, 'manifests': manifests,
                   'rng': rng_state(), 'loader_rng': generator.get_state(), 'history': history}
        save_checkpoint(payload, output / 'last.pt')
        if improved:
            save_checkpoint(payload, output / 'best.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'Epoch {epoch+1}/{args.epochs}: train acc={train["accuracy"]:.4f} F1={train["macro_f1"]:.4f}; val acc={val["accuracy"]:.4f} F1={val["macro_f1"]:.4f}')

if __name__ == '__main__':
    main()
