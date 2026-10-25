# ViT 训练调整

旧实验 runs/vit 保留。第 5 轮验证准确率 20%，所有样本预测为 FaceSwap。优化器有 17997 次更新（18000 批），AMP 跳过 3 次，排除完全没有更新的情况；尚不能确定塌缩的唯一原因。

新实验 runs/vit_adjusted：从 ImageNet 预训练权重重新初始化，结构不变，仍可用于融合。
- 前 2 轮固定 ViT，仅训练分类头。
- 之后联合训练：主干学习率 1e-6，分类头 3e-4。
- batch-size=1，accumulation-steps=8；当前完整数据恰好整除，有效 batch=8 段视频。
- 保持 16 帧、224 输入、原预处理、CUDA AMP、60% 近似计算占空比和显存分配上限。
- 先固定抽取官方 train 的 40 对视频、val 的 20 对视频，得到 400 个训练视频与 200 个验证视频，试跑 5 轮。测试集不参与。
- 子集最佳验证 accuracy>25%、macro F1>20%、至少预测三类才允许从头完整训练 30 轮，否则停在 needs_review。此门槛只是学习正常性检查，不代表已达到理想性能。

新增参数 --backbone-lr、--warmup-epochs、--accumulation-steps 均保存并随 resume 恢复。原有 checkpoint 使用旧优化器配置保持兼容。记录每轮 optimizer_updates 与 skipped_updates；best.pt 仍按验证 macro F1 保存。

恢复某个新实验时沿用其 splits-dir、output 与 last.pt，不直接重跑调度脚本。原始 ViT 模型及融合权重格式不变。没有更改人脸裁剪或新增人脸检测。
