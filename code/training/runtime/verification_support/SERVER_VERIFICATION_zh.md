Historical documentation with machine paths redacted. Original bytes remain in the scientific archive.
This document is not a valid signed execution plan; use the portable entry and a fresh prepare.

# 不重训的服务器补强与验证

本次使用现有 `dopnet_additions` 中的模型、636 条件评分和原实验协议，不训练神经网络，也不修改原模型、评分或验证集。`dopnet_frozen_observations` 已删除不影响执行：包内的冻结源代码按原条件和种子流式重放接收观测，每条条件全部 512 帧通过原六张量波形 SHA-256 检查后，才发布新传统算法分数。不会长期保存整套测试波形。

## 直接执行

将 `dopnet_no_training_verification_20261001.zip` 上传到 `__HISTORICAL_MACHINE_PATH_REDACTED__`，在原训练环境执行：

```bash
cd __HISTORICAL_MACHINE_PATH_REDACTED__ && unzip -oq dopnet_no_training_verification_20261001.zip && cd dopnet_verification && python run_verification.py --reference-results __HISTORICAL_MACHINE_PATH_REDACTED__ --output __HISTORICAL_MACHINE_PATH_REDACTED__ --suite manuscript
```

如果服务器实际结果位于双层同名工程目录，请只改 `--reference-results`。该参数也接受 `.../dopnet_additions/cells`，但其父目录必须保留 `experiment.json`、`evaluation.json`、`checkpoints/`。只上传 cells 无法完成模型时延测量。12 个现有 `model.pt`（DOP-Net、DOP-Net-Lite、Slow-time、FortiTran task adapter，各三个种子）由原结果目录读取，不需要重复打包或训练。

沿用当时生成原测试数据的 Python、PyTorch、CUDA 与 Sionna 环境，不要为了本次验证升级它们。先运行下面的元数据检查也可以：

```bash
python run_verification.py --reference-results __HISTORICAL_MACHINE_PATH_REDACTED__ --output __HISTORICAL_MACHINE_PATH_REDACTED__ --suite manuscript --stage preflight
```

默认 `manuscript` 是原主 NMSE 的 236 个条件与高移动 16-QAM BER 的 33 个条件，共 269 个条件。`core` 为原传统比较的 99 个条件；`all` 为全部 636 个条件。只需覆盖论文主要结论时使用默认范围。不同范围或参数应使用不同的 `--output`，不要复用已锁定结果目录。

## 执行内容与设备选择

流程依次运行：元数据检查 → 独立校准观测 → CPU/GPU 初测 → 完整校准 → 选定参数后的 CPU/GPU 复测 → 数字接收链时延 → 新传统算法评分 → 统一汇总和打包。

- **Pilot covariance LMMSE**：在原始 comb 导频位置进行统计线性估计，归一化频率与时间二阶矩，使用一致的导频噪声单位。默认按公开 CP/numerology 分组，384 帧独立训练混合校准内部以完整生成 chunk 分离拟合与留出；噪声比例选定后再用该组全部校准帧拟合。它是经验可分离协方差参考，不是 oracle。原历史 Wiener 平滑器分数与它分别命名。
- **Joint OCE-BEM、Joint DPSS-BEM**：保留原独立 128 帧校准，完整比较 Q=9/15/21、延迟支持 [-4,28) / [-8,40)、ridge=0.01/0.1/1/10/100，共 60 个候选。使用直接 Cholesky 或等价观测域解，同时检查恢复系数的正规方程残差。失败候选不能被当作低性能候选丢弃。参数选定并锁定后才评测试。
- **公平 BER**：主比较采用同一双用户对角 MMSE 和原 Gray 解调约定。BEM 的 full-band ICI 检测另列 native BER；统计参考没有 native 接口，报告中不填零。
- **设备适配**：传统算法在同一独立观测、complex128 精度下测 CPU 线程 1/4/8 与 CUDA；只有数值检查合格的后端参与速度排序，最终选参后复测的策略实际用于评分。CUDA OOM 可退到同精度 CPU，记录每种后端处理的帧数，不缩减候选网格、不改变估计器。生成原波形仍使用原 CUDA/Sionna 后端，即使传统求解选择 CPU。
- **端到端数字接收时延**：现有四种模型的三个种子、B=1/8/32、公共对角及各自 native 模式分别测 CPU/GPU；CPU 比较 1/4/8 线程。捕获真正加噪后的模拟时域接收样本，包含 CP 去除、1024 点 FFT/子载波选择、所需 LS/ADD 前端、模型、均衡、解调及比特输出。另列占用频带输入和 device-resident 边界。host-origin 总时间包含输入和输出传输，报告 p50/p95、吞吐与内存。没有包含 RF/ADC 硬件时延。B32 OOM 不会偷偷拆小后记成完整 B32 时延。
- **结果统计**：沿用原测试帧和比特分母，重用已存神经评分，新增三种传统参考分数。按八帧生成簇做共享配对 bootstrap；学习方法共享三个训练种子的重采样。NMSE 在线性域聚合后转 dB，BER 按错误/比特总数计算。自动生成 CSV、95% 区间、配对差值、表格、曲线与 LaTeX 插入文件。

默认 CPU/GPU 时延测量重复 30 次。运行计时阶段时尽量让服务器没有其他重负载；程序不会关闭其他任务。GPU 是否更快必须以输出为准，不预设 GPU 胜出。

## 输出与断点恢复

输出根目录为 `__HISTORICAL_MACHINE_PATH_REDACTED__`：

| 路径 | 内容 |
|---|---|
| `classical_backend_benchmark.json`、`classical_device_policy.json` | 选定传统参数的后端实测和实际评分策略 |
| `calibration_lock.json`、校准候选/先验文件 | 完整网格、先验分组、选择和代码绑定 |
| `cells/`、`COMPLETE.json` | 通过原波形校验的新逐帧评分与完成标记 |
| `latency/digital_receiver_latency.json` | 完整数字链时延、数值检查、OOM与硬件 |
| `latency/inference_device_policy.json` | 按模型、batch、接收模式和输入边界的推理设备推荐 |
| `corrected_summary/` | 统一指标、区间、图表、论文插入文件与就绪状态 |
| `verification_results_for_paper.zip` | 自动生成的回传包；包含新评分、图表、校准证据和时延，不包含大波形数组 |

中断后在同一环境以**同一指令**恢复：完整且哈希匹配的校准候选和 cell 分数会复用。不完整 cell 重新计算；原模型不重训。若代码、参数网格、校准数量或套件变化，使用新输出目录。

只重新制作图表和回传包：

```bash
python run_verification.py --reference-results __HISTORICAL_MACHINE_PATH_REDACTED__ --output __HISTORICAL_MACHINE_PATH_REDACTED__ --suite manuscript --stage report
```

将 `verification_results_for_paper.zip` 放回本地 `dopnet/实验结果及数据/` 后，可依据实测数字替换正文的传统比较和时延表。当前论文 PDF 数字尚未用未经服务器验证的新实现结果替换。`corrected_summary/statistics.json` 中评分、数字延迟和传统验证成本三项都齐备时，`paper_ready` 才为 true。

## 在已有推理中使用设备策略

新传统评分流程已自动读取其设备策略。神经模型原评分直接复用；如需追加固定权重推理，可使用以下接口读取神经时延策略，传入同格式的观测数组（不是原 cells 分数）：

```python
from verification_extension.latency import resolve_device, infer_observed_batch

choice = resolve_device('__HISTORICAL_MACHINE_PATH_REDACTED__',
                        method='dopnet', batch_size=8, mode='common_diagonal', source='calibration')
prediction = infer_observed_batch(observation_arrays, checkpoint['cfg'], checkpoint['model'],
                                 device=choice['preferred_device'], method='dopnet',
                                 mode='common_diagonal', cpu_threads=choice['cpu_threads'])
```

检查点键以原 `model.pt` 的实际状态为准；接口可接受已构建的模型或其 state dict。策略验证代码、硬件与环境，设备改变后应重新计时；微批 OOM 回退会记录实际拆分，不修改估计器。不同模型、batch、模式与输入边界的策略不要混用。

本轮不重训能够补齐传统估计修正、固定帧比较、完整数字链时延和硬件选择。外部方法完整原训练配方、更多独立训练种子及 OTA/边缘硬件实验仍需要对应训练或设备，不能由此次重算替代。
