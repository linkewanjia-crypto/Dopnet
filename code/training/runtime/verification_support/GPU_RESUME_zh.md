Historical documentation with machine paths redacted. Original bytes remain in the scientific archive.
This document is not a valid signed execution plan; use the portable entry and a fresh prepare.

# 本轮继续：数据生成用 GPU，时延只测 GPU

本次直接选择现有 CUDA/Sionna 批量生成器，不增加 CPU/GPU 生成测速，也不做 CPU 模型时延。传统算法校准、完整参数网格和已完成的求解后端选择直接复用。没有模型重训。

原入口仍属于旧执行版本。当前进程如果还在 `digital-latency ... cpu`，在运行它的终端按一次 **Ctrl+C** 结束该轮计时，再应用新包；已经完成的校准文件不会被撤销。不要同时运行两个入口。

将 `dopnet_gpu_only_resume_20261001.zip` 上传至 `__HISTORICAL_MACHINE_PATH_REDACTED__`，在原训练环境执行：

```bash
cd __HISTORICAL_MACHINE_PATH_REDACTED__ && unzip -oq dopnet_gpu_only_resume_20261001.zip && cd dopnet_verification && python resume_gpu_verification.py --reference-results __HISTORICAL_MACHINE_PATH_REDACTED__ --output __HISTORICAL_MACHINE_PATH_REDACTED__ --suite manuscript
```

此包新增独立入口，保留原校准所绑定的所有 Python 文件原字节。它不会重跑 `prepare`、`benchmark`、`calibrate` 或 `benchmark-selected`。

执行顺序为：

1. **GPU 接收链时延**：四种已有模型、三个种子、batch 1/8/32，仅 CUDA。包含原频域观测到比特输出和实际模拟时域输入到比特输出两个边界；时域输入包含 CP 去除与 FFT。CPU 只承担进程控制、checkpoint 读取与输入/输出主机传输，不运行 CPU 模型推理测速。
2. **新传统算法评分**：原 GPU 生成器按原配置、原种子重放测试波形。保留原链路分块设置。每条条件完整 512 帧通过原波形指纹后才保存新算法结果。已有神经模型评分直接复用；传统求解器沿用之前已选定的配置，不再重复 CPU/GPU 测速。
3. **统一图表和回传包**：生成逐条件指标、配对区间、传统对比图表与仅含 GPU 数值的接收链时延表。

CPU/GPU 随机流可能不同，本次选择原 GPU 后端同时保持测试数据与旧分数的对应关系。生成器选择是针对现有工程的执行选择，不冒充已经做过跨后端速度实验。

新时延保存在 `dopnet_verification_results/latency_gpu/`，旧的未完成 CPU 时延记录保留在 `latency/`，不进入新论文时延表。`generation_execution.json` 记录数据生成选用 GPU 且没有进行 CPU/GPU 生成测速。

若以后只需恢复指标计算与图表阶段：

```bash
python resume_gpu_verification.py --reference-results __HISTORICAL_MACHINE_PATH_REDACTED__ --output __HISTORICAL_MACHINE_PATH_REDACTED__ --suite manuscript --from-stage evaluate
```

流程完成后的回传文件仍为：

```text
__HISTORICAL_MACHINE_PATH_REDACTED__
```

本机仅做小型合成合同检查及代码、压缩包清单检查，没有执行真实数据生成、模型推理或硬件计时。
