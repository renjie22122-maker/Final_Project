# 实验记录与解释

此页摘录毕业设计报告 `contents/results&discussion.tex` 中的最终对比表。原实验平台：Windows 11、Python 3.10.15、RTX 4090 24 GB、PyTorch 2.0.1 + cu118。机器可读数据见 `experiment_results.csv`。

| 模型 | 时间 s/图 | M→R FID | M→R DINO | R→M FID | R→M DINO |
| --- | ---: | ---: | ---: | ---: | ---: |
| CycleGAN | 0.01 | 118.42 | 0.7783 | 61.34 | 0.7290 |
| CUT | 0.01 | 123.60 | 0.6161 | 68.72 | 0.4738 |
| UNSB | 0.07 | 112.76 | 0.6022 | 57.55 | 0.4784 |
| CycleGAN-Turbo | 0.13 | 103.43 | 0.2500 | 58.71 | 0.1879 |

FID 越低越好，表中的 DINO 是报告定义的结构相似度，越高越好，不能与低值更好的 DINO distance 混用。

- 莫奈→照片方向，Turbo 的 FID 最低（103.43）。
- 照片→莫奈方向，UNSB 的 FID 最低（57.55）。
- CycleGAN 在两个方向的 DINO 结构相似度最高（0.7783 / 0.7290）。

原报告部分讨论文字称 Turbo 在两个方向的 FID 都最低，与表格不一致；本整理以表格为准。UNSB 的正确展开是 **Unpaired Neural Schrödinger Bridge**。

`src/unsb/models/sb_model.py` 可见 LPIPS/DINO 损失相关实现，`ncsn_networks.py` 可见 U-Net 生成器变体。这证明代码包含探索，不能仅凭文件存在认定每个开关都有完整的控制实验；尤其当前 LPIPS/DINO 的部分命令行参数声明被注释，重新训练前需核对配置。上述模型对比也不能当作某个单独改动的因果提升。

这些数值未在本次整理中重新测量；留存表格未完整说明统一计时边界、所有随机种子和样本划分。若重新报告耗时，应分别测量冷启动、热模型推理和端到端 Web 请求；当前应用每次请求会重载模型，不能宣称达到表中的端到端时延。
