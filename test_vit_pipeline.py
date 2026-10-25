"""Offline tests of the actual ViT-B/16; requires several GB temporary disk space."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import gc

import cv2
import numpy as np
import torch
from model.vit import VideoViT

class ViTTest(unittest.TestCase):
    def test_forward_backward(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = VideoViT(pretrained=False).to(device).train()
        x = torch.randn(1, 2, 3, 224, 224, device=device)
        with torch.autocast(device_type=device, enabled=device == 'cuda'):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1], device=device))
        loss.backward()
        self.assertEqual(tuple(logits.shape), (1, 5))
        grad = model.backbone.encoder.layers[0].self_attention.in_proj_weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        model.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(x), model(x.flip(1)), rtol=1e-4, atol=1e-5)
        del model, x, logits, loss, grad
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    def test_checkpoint_and_test_entry(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits = root / 'splits'
            splits.mkdir()
            for i, split in enumerate(['train', 'val', 'test']):
                a, b = f'{i*2:03}', f'{i*2+1:03}'
                (splits / f'{split}.json').write_text(json.dumps([[a, b]]))
                for label, name in enumerate(['Real', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']):
                    folder = root / ('original_sequences/youtube' if label == 0 else f'manipulated_sequences/{name}') / 'c23/videos'
                    folder.mkdir(parents=True, exist_ok=True)
                    for stem in ([a, b] if label == 0 else [f'{a}_{b}', f'{b}_{a}']):
                        writer = cv2.VideoWriter(str(folder / f'{stem}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (32, 32))
                        self.assertTrue(writer.isOpened())
                        writer.write(np.full((32, 32, 3), 20 + 40 * label, dtype=np.uint8))
                        writer.release()
            output = root / 'run'
            common = ['--data-root', str(root), '--splits-dir', str(splits), '--device', device,
                      '--batch-size', '2', '--output', str(output)]
            def run(script, extra):
                p = subprocess.run([sys.executable, script, *common, *extra], capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            run('transformer_train.py', ['--epochs', '1', '--frames', '1', '--no-pretrained', '--freeze-backbone'])
            run('transformer_train.py', ['--epochs', '2', '--resume', str(output / 'last.pt')])
            history = json.loads((output / 'history.json').read_text())
            self.assertEqual(len(history), 2)
            self.assertEqual(history[-1]['val']['samples'], 10)
            run('transformer_test.py', ['--checkpoint', str(output / 'best.pt')])
            result = json.loads((output / 'test_metrics.json').read_text())
            self.assertEqual(result['samples'], 10)
            self.assertEqual(result['frames'], 1)
            self.assertEqual(len((output / 'predictions.csv').read_text(encoding='utf-8-sig').splitlines()), 11)

if __name__ == '__main__':
    unittest.main()
