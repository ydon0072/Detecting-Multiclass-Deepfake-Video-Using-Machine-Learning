"""Offline integration test with synthetic MP4s; never measures detection quality."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch

from config import CLASSES
from dataset.ffpp import FFPPDataset, read_splits, sample_indices
from utils.metrics import Metrics

class PipelineTest(unittest.TestCase):
    def test_metrics_and_sampling(self):
        m = Metrics()
        labels = torch.arange(5)
        m.update(torch.eye(5), labels, 0.5)
        self.assertEqual(m.compute()['accuracy'], 1)
        self.assertEqual(m.compute()['macro_f1'], 1)
        self.assertEqual(sample_indices(1, 4, True), [0, 0, 0, 0])
        m = Metrics()
        m.update(torch.tensor([[0., 1., 0., 0., 0.]]), torch.tensor([0]), 1.)
        self.assertEqual(m.compute()['macro_f1'], 0)

    def test_training_resume_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splitdir = root / 'splits'
            splitdir.mkdir()
            for i, split in enumerate(['train', 'val', 'test']):
                a, b = f'{2*i:03}', f'{2*i+1:03}'
                (splitdir / f'{split}.json').write_text(json.dumps([[a, b]]))
                for label, name in enumerate(CLASSES):
                    folder = root / ('original_sequences/youtube' if label == 0 else f'manipulated_sequences/{name}') / 'c23/videos'
                    folder.mkdir(parents=True, exist_ok=True)
                    for stem in ([a, b] if label == 0 else [f'{a}_{b}', f'{b}_{a}']):
                        writer = cv2.VideoWriter(str(folder / f'{stem}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (40, 40))
                        self.assertTrue(writer.isOpened())
                        for frame in range(3):
                            writer.write(np.full((40, 40, 3), label * 40 + frame * 5, dtype=np.uint8))
                        writer.release()
            splits = read_splits(splitdir)
            ds = FFPPDataset(root, splits['val'], frames=4, image_size=32)
            self.assertEqual(ds[0][0].shape, (4, 3, 32, 32))
            self.assertTrue(torch.equal(ds[0][0], ds[0][0]))
            common = [sys.executable, 'cnn_lstm_train.py', '--data-root', str(root), '--splits-dir', str(splitdir),
                      '--frames', '2', '--image-size', '32', '--hidden-size', '8', '--batch-size', '5',
                      '--no-pretrained', '--device', 'cuda' if torch.cuda.is_available() else 'cpu']
            def run(extra):
                result = subprocess.run(common + extra, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / 'run'
            run(['--epochs', '1', '--output', str(output)])
            first = torch.load(output / 'last.pt', weights_only=False, map_location='cpu')
            run(['--epochs', '2', '--resume', str(output / 'last.pt'), '--output', str(output)])
            resumed = torch.load(output / 'last.pt', weights_only=False, map_location='cpu')
            self.assertEqual(resumed['epoch'], 1)
            self.assertEqual(len(resumed['history']), 2)
            self.assertFalse(torch.equal(first['model']['classifier.1.weight'], resumed['model']['classifier.1.weight']))
            run(['--evaluate', 'test', '--resume', str(output / 'best.pt'), '--output', str(output)])
            self.assertEqual(json.loads((output / 'test_metrics.json').read_text())['samples'], 10)
            result = subprocess.run([sys.executable, 'cnn_lstm_test.py', '--data-root', str(root),
                                     '--splits-dir', str(splitdir), '--checkpoint', str(output / 'best.pt'),
                                     '--output', str(output / 'standalone'), '--batch-size', '5',
                                     '--device', 'cuda' if torch.cuda.is_available() else 'cpu'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            standalone = json.loads((output / 'standalone/test_metrics.json').read_text())
            previous = json.loads((output / 'test_metrics.json').read_text())
            self.assertEqual(standalone['confusion_matrix'], previous['confusion_matrix'])
            self.assertAlmostEqual(standalone['loss'], previous['loss'], places=5)
            self.assertEqual(len((output / 'standalone/predictions.csv').read_text(encoding='utf-8-sig').splitlines()), 11)
            # Continuous and resumed runs must follow the same trajectory.
            continuous = root / 'continuous'
            run(['--epochs', '2', '--output', str(continuous)])
            full = torch.load(continuous / 'last.pt', weights_only=False, map_location='cpu')
            for key in resumed['model']:
                torch.testing.assert_close(resumed['model'][key], full['model'][key], rtol=1e-5, atol=1e-6)
            (splitdir / 'test.json').write_text((splitdir / 'train.json').read_text())
            with self.assertRaisesRegex(ValueError, 'leakage'):
                read_splits(splitdir)

if __name__ == '__main__':
    unittest.main()

