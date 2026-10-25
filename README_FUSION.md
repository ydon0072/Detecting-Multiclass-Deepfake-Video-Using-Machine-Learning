# CNN+LSTM 与 ViT 注意力融合

## 冻结双分支实验

`run_fusion_frozen.py` 从 CNN+LSTM 第 20 轮和 ViT 第 9 轮最佳模型初始化，仅训练投影、注意力、池化打分和 MLP。两分支的参数及 BatchNorm 统计量保持不变。输出目录 `runs/fusion_frozen`。

采用 16 帧、batch-size=2、梯度累积 4 次（有效 batch=8）、AdamW 学习率 1e-4、weight decay 1e-4、最多 30 轮，连续 5 轮验证 macro F1 无提高则停止。CUDA AMP、gpu-duty=0.6 和 gpu-memory-fraction=0.6 延续先前负载设置，实际设备利用率不是精确上限。仅使用官方 train/val，暂不运行测试集或自动解冻。

`status.json` 保存进程状态；`history.json` 保存每轮指标；`best.pt` 按验证 macro F1 选择，`last.pt` 供恢复。不得重复启动已有实验。恢复示例：

```powershell
& 'E:\python\python.exe' fusion_train.py --data-root 'E:\study\论文代码\data\FaceForensics++' --splits-dir data/splits --resume runs/fusion_frozen/last.pt --output runs/fusion_frozen --epochs 30 --gpu-duty 0.6 --gpu-memory-fraction 0.6
```

同一段视频、同一组帧输入两个分支，沿用原先数据预处理和固定五分类顺序：

1. CNN+LSTM：ResNet18 每帧特征 → LSTM 最后隐藏状态，默认 256 维。
2. ViT：ViT-B/16 的每帧 CLS 特征 → 时间平均，768 维。
3. 各自线性投影、LayerNorm 到 fusion_dim=256，叠为两个 branch token，并加可学习分支 embedding。
4. 4-head self-attention、残差连接、LayerNorm，使两个 token 交换信息。
5. 共享 MLP 给 token 打分，softmax 后加权求和，得到融合特征。
6. 最终 MLP 分类头：Linear(256→128) → LayerNorm → GELU → Dropout(0.3) → Linear(128→5)。

最终分类头与融合模块一起训练，输出 `[batch_size, 5]` 原始 logits，交叉熵损失直接接收 logits；测试时 softmax 转成五类概率，再用 argmax 得到预测类别。LayerNorm 支持 batch-size=1。可用 `--mlp-hidden 128 --mlp-dropout 0.3` 配置；配置写入 checkpoint，resume 和测试自动恢复。

模型架构标识更新为 `cnn_lstm_vit_attention_mlp_v2`。此前只有线性头的融合 checkpoint 不可直接 resume 到此版本；两个原始分支的 checkpoint 仍可用于初始化。分类头可以输出最终结果，但真实准确率必须在完成融合训练后通过独立测试得到，增加 MLP 不保证准确率提高。

这是视频级特征融合，不是概率平均，也不是逐帧跨注意力。ViT 分支本身仍不编码时间顺序。`model(x, return_attention=True)` 可返回 `[B,2]` 池化权重及 `[B,heads,2,2]` 自注意力矩阵；token 顺序为 CNN+LSTM、ViT。这些权重对应交互后的特征，不应直接解释为因果贡献。

## 文件

`model/attention_fusion.py`、`fusion_config.py`、`utils/fusion_setup.py`、`fusion_train.py`、`fusion_test.py`、`test_fusion_pipeline.py`。不修改已有单分支代码。依赖沿用 requirements.txt。

## 当前已有 CNN+LSTM checkpoint，尚无训练好的 ViT checkpoint

```powershell
cd 'E:\study\论文代码'
& 'E:\python\python.exe' fusion_train.py --data-root 'E:\study\论文代码\data\FaceForensics++' --splits-dir data/splits --cnn-checkpoint runs/branch1_experiment/full/best.pt --frames 16 --batch-size 1 --lr 0.00003 --epochs 30 --output runs/fusion
```

该命令加载已有 CNN+LSTM 模型，用 ImageNet 预训练初始化 ViT，联合微调两个分支与融合层。首次需要联网下载未缓存的权重。没有分支 checkpoint 时默认从 ImageNet backbone 初始化；LSTM 和融合层随机初始化。`--no-pretrained` 用于离线流程测试。原单分支分类头不参与融合，也不更新。

如果以后两个分支都训练完成，可添加 `--vit-checkpoint runs/vit/best.pt`。两个 checkpoint 的类别、划分哈希、帧数、压缩等级和图像尺寸必须匹配当前实验，CNN 隐藏维度也须一致。若参数不同，明确修改配置，代码不会静默忽略。

两个 checkpoint 都提供后，`--freeze-branches` 可固定整个分支，只训练投影、注意力和融合分类头；缺少训练好的分支时禁止该选项，避免误冻随机 LSTM。默认联合微调。若想切换冻结策略，开始新的训练实验；resume 恢复原训练策略。

## 续训和测试

```powershell
& 'E:\python\python.exe' fusion_train.py --data-root 'E:\study\论文代码\data\FaceForensics++' --splits-dir data/splits --resume runs/fusion/last.pt --epochs 50 --output runs/fusion
& 'E:\python\python.exe' fusion_test.py --data-root 'E:\study\论文代码\data\FaceForensics++' --checkpoint runs/fusion/best.pt --output runs/fusion/test
```

resume 加载完整融合模型、优化器、AMP scaler 和随机状态，不再需要原分支 checkpoint；保持同一 output 目录以保留最佳模型。epochs 为总轮数。每轮保存 last.pt，验证集 macro F1 提升时保存 best.pt。测试只读取 FF++ test，输出 accuracy、macro F1、各类 precision/recall/F1、预测 CSV 和混淆矩阵。测试集不用于调参。

输入固定 224×224；AMP 下两个 backbone 混合精度，LSTM 和融合层 float32。联合训练显存需求高于单分支，命令从 batch-size=1 起步。CPU 调试加 `--device cpu`，Windows 初始 workers=0，可尝试 2。

```powershell
& 'E:\python\python.exe' -m unittest test_fusion_pipeline -v
```

测试包含实际双分支的 GPU 梯度、权重归一化、合成 MP4 训练/保存/恢复/测试；不使用真实训练数据。临时 checkpoint 需要数 GB 磁盘空间。编写代码后不会自动启动真实融合训练。
