# LeJEPA 独立复现

[English](README.md) | **简体中文**

本文是 *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*
（Balestriero & LeCun，[arXiv:2511.08544](https://arxiv.org/abs/2511.08544)）的独立实现。

全部代码从零编写，未 import 官方 `lejepa` 包的任何实现。Epps–Pulley 统计量、SIGReg、随机切片、
Prediction Loss 与训练循环均依据论文公式重新推导，并用解析结果做了数值校验。

方法复现完成；论文在 ImageNet-1K 上的结果未复现，原因见「复现范围」。

---

## 复现范围

| | |
|---|---|
| **已复现** | 目标函数与训练配方；统计模块，已对照解析解验证；ImageNet-100 上的端到端预训练 |
| **未复现** | ImageNet-1K、ViT-L/14 与 ViT-H/14、论文报告的各项精度 |

未运行 ImageNet-1K 的原因是磁盘而非算力：训练集压缩包为 147.9 GB，而复现主机唯一可写卷为 50 GB。
ViT-L/14 在 batch 512 × 8 视图 × 238 px 下约需 60 GB 激活显存，24–32 GB 显卡无法容纳。
详细说明与三级分类见 [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md)。

## 结果

ImageNet-100，ViT-S/16，bf16，AdamW，线性 warmup + cosine 衰减。评估方式为冻结主干后接线性探针，
输入取最后两层 CLS 的拼接，与论文 README 的协议一致。

| 运行 | 视图数 | λ | Epochs | 线性探针 | k-NN |
|---|---|---|---|---|---|
| **Exp C** | 8（2×224 + 6×96） | 0.05 | 50 | **69.02 %** | 60.24 % |
| **Exp B** | 4（128 px） | 0.02 | 40 | **49.52 %** | 42.78 % |

Exp B 的 SIGReg 由 9.89 收敛至 **1.54**，各向同性高斯分布对应的理论下界为 1.0594。
投影器空间的有效秩由 8.3 升至 15.8（满分 16），所有配置正确的运行均未出现表征坍塌。

消融实验每组 12 epoch，运行间噪声约 ±2–3 个点（见 [docs/ABLATIONS.md](docs/ABLATIONS.md)）：

| 变体 | 线性探针 |
|---|---|
| 基线，λ = 0.02 | 30.9 % |
| λ = 0.05 | 30.4 % |
| M = 64 切片 | 28.6 % |
| λ = 0.005 | 27.2 % |
| λ = 0.2 | 4.9 % |
| V = 2 视图 | 3.2 % |

后两组差异远超噪声，结论可靠。λ = 0.2 与 V = 2 都表现为嵌入分布仍为正确的高斯、但视图不变性被破坏，
这正是 λ 权衡所控制的失效模式。前四组的差异落在噪声范围内，不应据此排序。

八次运行的权重已作为 Release 资产发布，说明见 [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md)。

## 方法

| | |
|---|---|
| Epps–Pulley 统计量 | $N\int \lvert\hat\varphi_X(t)-\varphi(t)\rvert^2 w(t)\,dt$，其中 $w(t)=e^{-t^2/2}$ |
| SIGReg | $\frac{1}{\lvert\mathcal A\rvert}\sum_{a\in\mathcal A}\mathrm{EP}(\{a^\top z_n\})$，方向 $a\in S^{K-1}$ 每步重采样 |
| Prediction Loss | $\frac{1}{V}\sum_{v'}\lVert\mu_n-z_{n,v'}\rVert^2$，其中 $\mu_n=\frac{1}{V_g}\sum_{v\le V_g}z_{n,v}$ |
| 总目标 | $\lambda\,\mathrm{SIGReg}+(1-\lambda)\,\mathcal L_{\text{pred}}$ |

实现中不含 stop-gradient、EMA teacher、predictor，也不做梯度裁剪。
`tests/test_invariance.py::test_no_stop_gradient_every_view_receives_gradient` 断言每个视图
都能收到非零梯度。

## 目录结构

```
losses/       epps_pulley.py  sigreg.py  invariance.py
models/       backbone.py（timm ViT，动态图像尺寸）  projector.py（MLP，视图重排）
datasets/     transforms.py（多裁剪增强）  imagenet.py（视图数据集与 DataLoader）
engine/       trainer.py  optimizer.py  distributed.py
evaluation/   linear_probe.py（冻结探针 + 加权 k-NN）
scripts/      train.py  evaluate.py  prepare_in100.py  ablation_statistics.py
configs/      debug  imagenette  in100_vits16_128  imagenet100_vits16  imagenet_vit
tests/        五个文件，共 78 项测试
docs/         AUDIT.md  ABLATIONS.md  DATASETS.md  ENVIRONMENT.md  RELEASE_NOTES.md
```

## 安装

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install pytest scipy          # 仅测试需要
```

已在 Python 3.12.3、torch 2.8.0+cu128、torchvision 0.23.0、timm 1.0.30、RTX 4080 SUPER 上验证。
环境细节见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

## 数据准备

公开的 ImageNet-100 压缩包只含 126,689 张训练图，不含验证集。验证集需从 ImageNet-1K 验证集构造：

```bash
python scripts/prepare_in100.py \
  --raw     /path/to/in100_raw/imagenet100 \
  --out     /path/to/imagenet100 \
  --val-tar /path/to/ILSVRC2012_img_val.tar \
  --devkit  /path/to/ILSVRC2012_devkit_t12.tar.gz
```

输出为 100 个类别、126,689 张训练图与 5,000 张验证图。类别索引通过 devkit 解析，
不依赖任何硬编码的 wnid 顺序。

## 训练

```bash
# 合成数据上的流程自检：前向、反向、优化步、存档、恢复
python scripts/train.py --config configs/debug.yaml --dry-run

# Exp B：4 视图，128 px
python scripts/train.py --config configs/in100_vits16_128.yaml

# Exp C：8 视图，论文配方
python scripts/train.py --config configs/imagenet100_vits16.yaml --set epochs=50 batch_size=192

# 断点恢复、单项覆盖、四卡训练
python scripts/train.py --config configs/in100_vits16_128.yaml --resume auto
python scripts/train.py --config configs/in100_vits16_128.yaml --set lambd=0.05
torchrun --nproc_per_node=4 scripts/train.py --config configs/imagenet_vit.yaml
```

每次运行会写出 `config.json`、`metrics.jsonl`、`metrics.csv`、TensorBoard 目录与周期性 checkpoint。
checkpoint 会一并保存模型、优化器、学习率调度器、SIGReg 方向生成器的步数计数器与 RNG 状态，
因此恢复后的训练会延续同一条随机方向序列。

`configs/imagenet_vit.yaml` 完整记录了论文的 ImageNet-1K 配置，保留以备参照，运行需要本项目不具备的硬件。

## 评估

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/in100_vits16_128/checkpoints/final.pt \
  --data-root /path/to/imagenet100 --data-name imagenet100 \
  --num-classes 100 --resolution 128 --feature-source last2cls \
  --probe-epochs 100 --batch-size 256
```

`last2cls` 对应论文协议（最后两层 CLS 拼接，探针内含 LayerNorm，AdamW 权重衰减 1e-6）；
`embedding` 则改为探测主干 head 的输出。两种协议同时给出加权 k-NN 结果（k = 20，余弦相似度，T = 0.07）。

## 测试

```bash
python -m pytest tests/ -v
```

共 78 项。测试校验的是数值而非形状：

| 测试 | 所验证的结论 |
|---|---|
| `test_constant_sample_matches_quadrature_reference` | 统计量与自适应数值积分一致，相对误差 2e-4 |
| `test_mmd_identity_random_samples` | $\mathrm{EP}/N=\sqrt{2\pi}\,\mathrm{MMD}^2$ 的高斯核闭式关系 |
| `test_standard_normal_expectation_is_exact_constant` | 零假设下 $E[\mathrm{EP}]=\sqrt{2\pi}(1-1/\sqrt3)=1.0594$ |
| `test_collapse_is_a_saddle_not_an_attractor` | 完全坍塌点梯度为零且为局部极大值 |
| `test_paper_eq5_equals_eq7_plus_global_variance` | 修正后的式 (5) → (7) 恒等式 |
| `test_recovers_isotropic_gaussian_from_adversarial_x` | 论文图 6 的合成恢复实验 |
| `test_peak_lr_equals_configured_lr` | 一处会把学习率缩小 100 倍的调度器回归 |

## 关于论文的两点说明

实现过程中发现两处问题，连同论文与官方代码的另外八处不一致，一并记录在
[docs/AUDIT.md](docs/AUDIT.md)：

1. **式 (5) 与式 (6)/(7) 并不相等。** 附录 B.6 断言
   $\frac{1}{V_g}\sum_v\lVert z_v\rVert^2=\lVert\bar z\rVert^2$，该式仅在各 global view
   已经完全相同时成立。正确的恒等式为
   $\text{Eq.}(5)=\text{Eq.}(7)+\frac{1}{V_g}\sum_{v\le V_g}\lVert z_v-\bar z\rVert^2$。
   Algorithm 2 实现的是式 (7)，因此本项目按式 (7) 训练。
2. **完全坍塌点是不稳定鞍点，而非吸引子。** Epps–Pulley 统计量在该点梯度恒为零，
   但任何扰动都会使其下降，故坍塌是不稳定的。这正是 SIGReg 防止表征坍塌的机制。

另外观察到方法本身的一项局限：Rademacher ±1 嵌入在 D = 2 时统计量为 97.3，在 D = 256 时降至 1.02，
与正确嵌入无法区分。乘积分布的一维投影依中心极限定理趋于高斯，因此在高维下各方向边缘分布都正确，
联合分布却未必。SIGReg 约束的是边缘分布，不是联合分布。

## 已知局限

- 未运行 ImageNet-1K，未使用 ViT-L/14 或 ViT-H/14，因此本仓库的任何数值都无法与论文报告的精度直接对比。
- Exp B 与 Exp C 在数据集、patch 大小、batch 与 epoch 预算上均不同，
  两者之间 19.5 个点的差距不能单独归因于视图数量。
- 消融每组仅 12 epoch，只支持量级层面的结论。
- 坍塌诊断监控的是投影器输出，而评估读取的是主干特征。主干空间的诊断脚本
  （`scripts/diagnose_checkpoints.py`）已写出但未完成，原因是当时可用的 CPU 配额只有 0.5 核。

## 许可

MIT，见 [LICENSE](LICENSE)。

## 引用

```bibtex
@misc{balestriero2025lejepa,
  title  = {LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author = {Balestriero, Randall and LeCun, Yann},
  year   = {2025},
  eprint = {2511.08544},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
