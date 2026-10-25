# 第二分支：ViT 视频五分类

`transformer_train.py` 与 `transformer_test.py` 使用 `model/vit.py` 的 VideoViT，配置独立放在 `vit_config.py`。

结构：视频抽帧 → 每帧 ViT-B/16 → 768 维 CLS 特征 → 对帧取平均 → Dropout + Linear → 五类 logits。
不使用 CNN backbone、LSTM 或额外时序 Transformer。torchvision 内部的 Conv2d 是 ViT 的标准 patch 投影实现。此模型对帧排列顺序不敏感，不能描述为显式学习时间顺序的视频 Transformer。

默认采用 ImageNet 预训练权重，首次训练需要联网下载。恢复/测试不会重新下载。输入固定 224×224；归一化和中心裁剪沿用 CNN+LSTM 数据管线，便于比较（并非 torchvision 权重默认 resize 256 的完整推理预处理）。

## 运行（PowerShell）

```powershell
cd 'E:\study\论文代码'
& 'E:\python\python.exe' transformer_train.py --data-root 'E:\datasets\FaceForensics++' --splits-dir data/splits --epochs 30 --batch-size 2 --frames 16 --output runs/vit
& 'E:\python\python.exe' transformer_train.py --data-root 'E:\datasets\FaceForensics++' --splits-dir data/splits --resume runs/vit/last.pt --epochs 50 --output runs/vit
& 'E:\python\python.exe' transformer_test.py --data-root 'E:\datasets\FaceForensics++' --checkpoint runs/vit/best.pt --output runs/vit/test
```

替换真实数据根目录。依赖沿用 requirements.txt，无新增包。默认 GPU + AMP、workers=0，完整训练 ViT。显存不足时先降 batch-size 到 1；再降 frames，但应记录与另一分支的采样差异。`--freeze-backbone` 仅训练分类头，是不同训练设置；`--no-pretrained` 可离线自检。CPU 可使用 `--device cpu`。

使用 FF++ 官方 train/val/test 划分，只用 val macro F1 选择 best.pt，test 不参与调参。checkpoint 带 ViT 架构标识，错误分支权重会被拒绝。续训沿用原输出目录，恢复模型、优化器、GradScaler、随机状态、训练配置；epochs 是总目标数，恢复到上一完整 epoch。

训练输出 last.pt、best.pt、history.json、config.json、manifest.json。测试输出 test_metrics.json、predictions.csv、confusion_matrix.csv，与 CNN+LSTM 的指标定义一致。正常完整 test 共 700 段，每类 140 段。暂不使用 Celeb-DF，不进行人脸检测。

```powershell
& 'E:\python\python.exe' -m unittest test_vit_pipeline -v
```

自检实际 ViT 的 GPU 前后向、帧平均顺序不变性，以及合成视频的训练/恢复/测试流程；流程测试冻结 backbone 以减少开销，另有独立测试验证 backbone 梯度。临时 checkpoint 需数 GB 磁盘空间。合成测试不代表检测性能。

参考：[torchvision ViT-B/16 官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html)。
