import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import cv2
import numpy as np
import torch
from model.attention_fusion import AttentionFusion

class FusionTest(unittest.TestCase):
    def test_attention_and_gradients(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AttentionFusion(hidden_size=8, fusion_dim=16, heads=4, pretrained=False).to(device)
        x = torch.randn(1, 2, 3, 224, 224, device=device)
        with torch.autocast(device_type=device, enabled=device == 'cuda'):
            logits, attention = model(x, return_attention=True)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1], device=device))
        loss.backward()
        self.assertEqual(tuple(logits.shape), (1, 5))
        torch.testing.assert_close(attention['branch_weights'].sum(1), torch.ones(1, device=device))
        for param in [model.cnn_lstm.backbone.conv1.weight, model.cnn_lstm.lstm.weight_ih_l0,
                      model.vit.backbone.encoder.layers[0].self_attention.in_proj_weight,
                      model.attention.in_proj_weight, model.score[-1].weight,
                      model.classifier[0].weight, model.classifier[-1].weight]:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(), 0)
        self.assertIsNone(model.cnn_lstm.classifier[-1].weight.grad)
        del model, x, logits, attention, loss, param
        gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()

    def test_train_and_test(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = root / 'splits'
            splits.mkdir()
            for i, split in enumerate(['train', 'val', 'test']):
                a, b = f'{2*i:03}', f'{2*i+1:03}'
                (splits / f'{split}.json').write_text(json.dumps([[a,b]]))
                for label, name in enumerate(['Real','Deepfakes','Face2Face','FaceSwap','NeuralTextures']):
                    directory = root / ('original_sequences/youtube' if not label else f'manipulated_sequences/{name}') / 'c23/videos'
                    directory.mkdir(parents=True, exist_ok=True)
                    for stem in ([a,b] if not label else [f'{a}_{b}', f'{b}_{a}']):
                        writer = cv2.VideoWriter(str(directory / f'{stem}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (32,32))
                        self.assertTrue(writer.isOpened())
                        writer.write(np.full((32,32,3),20+40*label,dtype=np.uint8))
                        writer.release()
            output = root / 'run'
            common = ['--data-root',str(root),'--splits-dir',str(splits),'--device',device,'--output',str(output)]
            def run(script, args):
                result = subprocess.run([sys.executable,script,*common,*args],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            run('fusion_train.py',['--epochs','1','--frames','1','--hidden-size','8','--fusion-dim','16','--mlp-hidden','32','--mlp-dropout','0.2','--no-pretrained'])
            run('fusion_test.py',['--checkpoint',str(output/'best.pt')])
            result = json.loads((output/'test_metrics.json').read_text())
            self.assertEqual(result['samples'],10)
            self.assertTrue((output/'predictions.csv').exists())
            run('fusion_train.py',['--epochs','2','--resume',str(output/'last.pt')])
            self.assertEqual(len(json.loads((output/'history.json').read_text())),2)

if __name__ == '__main__':
    unittest.main()
