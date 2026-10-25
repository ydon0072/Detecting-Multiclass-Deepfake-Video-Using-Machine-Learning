# FF++ CNN + LSTM 五分类 baseline

## 独立 FF++ 测试入口

训练入口已更名为 `cnn_lstm_train.py`，独立测试使用 `cnn_lstm_test.py`：

```powershell
cd 'E:\study\论文代码'
& 'E:\python\python.exe' cnn_lstm_test.py --data-root 'E:\datasets\FaceForensics++' --splits-dir data/splits --checkpoint runs/cnn_lstm/best.pt --output runs/cnn_lstm/test
```

请替换数据路径，并先完成训练以生成 best.pt。测试只读取官方 test 划分，自动从 checkpoint 恢复帧数、图像尺寸、压缩等级和模型结构，不更新权重、不下载预训练权重。默认 CUDA + AMP，CPU 可加 `--device cpu`。每个视频均匀采样一个 clip，不使用随机增强。

输出 `test_metrics.json`（loss、accuracy、macro F1、macro precision/recall 和各类指标）、`predictions.csv`（视频路径、真实类别、预测类别、五类概率）、`confusion_matrix.csv`（行真实、列预测）。缺少某类别的预测时该类别 precision 记为 0。正常完整 FF++ test 应为每类 140 段、共 700 段视频。

各 branch 先统一使用 FF++ 的 train/val/test 划分和预处理；用 val 选择模型及超参数，最终用 test 报告五分类结果。该实验衡量同数据集表现，不能据此声称跨数据集泛化；Celeb-DF 可在后续独立实验加入。

使用 ResNet18 提取每帧 512 维特征，再通过单层 LSTM 和分类头输出一个视频的 5 类 logits。
固定类别顺序：`0 Real, 1 Deepfakes, 2 Face2Face, 3 FaceSwap, 4 NeuralTextures`。

## 文件结构

```text
config.py                命令行参数与类别顺序
dataset/ffpp.py           官方划分、视频抽帧、归一化
model/cnn_lstm.py         ResNet18 + LSTM
utils/metrics.py         视频级 accuracy、macro F1、混淆矩阵
utils/runtime.py         随机种子与 checkpoint
cnn_lstm_train.py                 训练、验证、恢复、测试
download_splits.py       下载官方划分文件
test_pipeline.py         合成视频集成测试
requirements.txt
```

## 1. Python / GPU 环境

本机已检测到 `E:\python\python.exe`，PyTorch `2.6.0+cu124`、torchvision `0.21.0+cu124`，CUDA 可用。
在 VS Code 的 **Python: Select Interpreter** 中选择此解释器。PowerShell：

```powershell
cd 'E:\study\论文代码'
& 'E:\python\python.exe' -m pip install -r requirements.txt
& 'E:\python\python.exe' -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

若换机器，请先从 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择匹配系统和驱动的 CUDA 版 torch/torchvision，再安装 requirements。不要在已有可用 GPU 环境中盲目升级。代码使用 `torch.amp.GradScaler` 和 `torch.autocast`。

## 2. 数据目录与官方划分

数据根目录示例 `E:\datasets\FaceForensics++`（请替换为自己的真实路径）：

```text
FaceForensics++/
  original_sequences/youtube/c23/videos/000.mp4
  manipulated_sequences/Deepfakes/c23/videos/000_003.mp4
  manipulated_sequences/Face2Face/c23/videos/000_003.mp4
  manipulated_sequences/FaceSwap/c23/videos/000_003.mp4
  manipulated_sequences/NeuralTextures/c23/videos/000_003.mp4
```

文件名仅为结构示例，具体配对由官方 JSON 决定。压缩级别全程一致，默认 c23；raw/c40 可通过 `--compression` 选择。不包含 DeepFakeDetection 类。

```powershell
& 'E:\python\python.exe' download_splits.py --output data/splits
```

该脚本仅下载官方 train/val/test JSON，不下载视频。也可手动从 [FF++ 官方划分](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits) 放入 `data/splits`。代码按原始视频 ID 配对划分，每个伪造类别包含两个方向；检查三组源视频 ID 不重叠。完整数据每类 train/val/test 为 720/140/140。默认缺文件即报错；仅有意使用小规模子集时加 `--allow-missing`，且每组仍须具备五类。

## 3. 训练

```powershell
& 'E:\python\python.exe' cnn_lstm_train.py --data-root 'E:\datasets\FaceForensics++' --splits-dir data/splits --epochs 30 --batch-size 2 --frames 16 --workers 0 --output runs/cnn_lstm
```

默认要求 CUDA，启用 AMP；CNN 使用混合精度，LSTM/分类头采用 float32。首跑会下载 ImageNet ResNet18 权重；离线流程测试可加 `--no-pretrained`，正式实验应记录是否使用预训练。默认整个 CNN 一同训练；`--freeze-backbone` 可固定 CNN 及其 BatchNorm 统计量、只训练 LSTM 和分类头。

每个视频按时间分段随机抽帧，验证/测试固定均匀抽帧。短视频重复帧补齐。整段视频共用随机水平翻转，缩放短边后中心裁剪，再使用 ImageNet mean/std 归一化。当前没有人脸检测或跟踪：这是先跑通的整帧/中心裁剪 baseline，中心外的人脸可能被裁掉。论文后续实验应统一人脸裁剪流程并记录预处理差异。

Windows 初始使用 workers=0；数据加载成为瓶颈时再尝试 2 或 4。显存不足先降低 batch-size，再降低 frames。CPU 调试使用 `--device cpu`，禁用混合精度使用 `--no-amp`。

## 4. 断点续训与测试

```powershell
& 'E:\python\python.exe' cnn_lstm_train.py --data-root 'E:\datasets\FaceForensics++' --splits-dir data/splits --resume runs/cnn_lstm/last.pt --epochs 50 --output runs/cnn_lstm
& 'E:\python\python.exe' cnn_lstm_train.py --data-root 'E:\datasets\FaceForensics++' --splits-dir data/splits --resume runs/cnn_lstm/best.pt --evaluate test --output runs/cnn_lstm
```

`--epochs 50` 是总计 50 个 epoch。恢复模型、优化器、GradScaler、随机状态和 DataLoader 随机状态，以及保存的模型/预处理/训练设置；从上一个已完成 epoch 继续，未完成 epoch 会重跑。恢复时检查划分和文件选择一致（不对视频内容做哈希）。跨硬件或软件版本不保证完全相同的数值结果。checkpoint 包含 Python/NumPy 状态，只加载自己信任的文件。

每个 epoch 写入 `last.pt`；验证集 macro F1 改善时写入 `best.pt`。请续训时沿用原 output 目录以保留此前 best.pt。`history.json` 含训练/验证 loss、accuracy、固定五类 macro F1、各类 F1 和混淆矩阵（行=真实、列=预测）。`config.json` 和 `manifest.json` 保存参数及数据清单。测试输出 `test_metrics.json`，测试集不参与 checkpoint 选择。一次验证/测试每个视频一个 clip，而非多 clip 集成。

## 5. 流程自检

```powershell
& 'E:\python\python.exe' -m unittest test_pipeline -v
```

测试生成临时 MP4，检查短视频采样、指标、CUDA AMP 训练（有 CUDA 时）、checkpoint、续训与连续训练一致性、测试集评估、划分泄漏检查，不下载预训练权重。合成数据结果不代表实际 deepfake 检测效果。

官方参考：[FF++ 数据说明](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md)、[ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)、[AMP](https://docs.pytorch.org/docs/stable/amp.html)。

