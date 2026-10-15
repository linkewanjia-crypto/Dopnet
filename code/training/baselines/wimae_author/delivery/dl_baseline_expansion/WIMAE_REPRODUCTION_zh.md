Historical documentation with machine paths redacted. Original bytes remain in the scientific archive.
This document is not a valid signed execution plan; use the portable entry and a fresh prepare.

# WiMAE 任务适配实验

这是独立的原始 WiMAE 架构适配：每个 OFDM 符号联合处理8天线×128子载波，保持12层编码器、4层解码器、维度64、16/8头和1×16 patch。它与旧 `wimae_ce`、PilotWiMAE 不同；不宣称原论文数据集或3000轮预训练的完整复现。

每个训练种子共40轮：1–20轮随机成对RI遮罩重建，21–40轮用真实公共LS/插值观测监督CE。每轮12,000 fresh multiuser frames；生成batch32、Sionna link chunk8保持原值。固定2,000帧历史验证数据只用于选择checkpoint，测试数据不进入训练或归一化。

官方最小源码和MIT许可位于 `vendor/wimae_upstream`，固定提交和各文件SHA在 `UPSTREAM.json`。局部package initializer是命名空间适配，其余六个实现文件保持下载字节不变。模型、配置和训练脚本有独立签名，原训练工程及运行计划不修改。

来源：[作者原始代码](https://github.com/BerkIGuler/WirelessContrastiveMaskedLearning/tree/c0909cf37c7d04f9598005da7f18d1501b8daa6c)、[原论文](https://arxiv.org/html/2505.09160v1)。

有效优化器batch固定32帧。默认梯度microbatch1帧，但每帧展开2用户×25时间片，即50个8×128 channel matrix。准备正式计划前，在服务器对microbatch1/2/4/8做两阶段合成计算探针，选速度最快且显存合适的候选；探针不产生真实波形，不读取验证数据，也不能单独证明波形生成并发安全。正式计划记录选定microbatch，不能热改。

精度默认BF16，参数/梯度与原始尺度CE MSE为FP32；也可签名配置为FP32。采用Adam、梯度裁剪1、每阶段20轮cosine，第二阶段重置优化器/调度器并从第一阶段最佳验证权重开始。20+20、batch32帧、BF16、pilot RMS和CE训练均为明示的任务适配。

仅在服务器运行，工作目录为含有 `dl_baseline_expansion` 的独立代码根目录。准备元信息示例：

```bash
python -m dl_baseline_expansion.prepare_wimae --project __HISTORICAL_MACHINE_PATH_REDACTED__ --reference-results __HISTORICAL_MACHINE_PATH_REDACTED__ --validation-cache __HISTORICAL_MACHINE_PATH_REDACTED__ --output __HISTORICAL_MACHINE_PATH_REDACTED__ --microbatch 1
```

探针示例（UTC截止时间必须替换为已授权预算的真实截止时间；不得因此延长预算）：

```bash
python -m dl_baseline_expansion.train_wimae --project __HISTORICAL_MACHINE_PATH_REDACTED__ --manifest __HISTORICAL_MACHINE_PATH_REDACTED__ --seed 42 --deadline-utc APPROVED_UTC_DEADLINE --probe-only --probe-microbatch 1 --probe-seconds 45 --probe-output __HISTORICAL_MACHINE_PATH_REDACTED__
```

使用选定microbatch另建正式输出后，逐种子运行同一训练入口，不带 `--probe-only`。所有实际训练必须有显式 `--deadline-utc`；时间不足时返回75并标记incomplete，只能从完整提交的上一轮继续。只有40轮全部提交、阶段记录完整及CE最佳模型与验证账本相符才算完成。

统一队列 `queue_wimae.py` 等待固定40轮FortiTran的队列与所有控制器/worker完全退出，并通过其交付包验证后，才串行测试microbatch1/2/4/8。每个探针Torch显存上限6GiB，预留34.5GiB供原波形生成并另留3GiB余量，同时核对实际total/free；所有候选不合格则停止，不降低generation32/link8。它选择两阶段合计吞吐最快的安全候选，并生成新的正式签名配置。每个阶段只按同一显式UTC截止时间运行。

```bash
python -m dl_baseline_expansion.queue_wimae --project ARCHIVED_PROJECT --reference-results REFERENCE_RESULTS --validation-cache HISTORICAL_VALIDATION_PT --output __HISTORICAL_MACHINE_PATH_REDACTED__ --forti-queue-state FORTI_TRAINING_OUTPUT/queue/queue_state.json --forti-training-output FORTI_TRAINING_OUTPUT --forti-evaluation FORTI_TRAINING_OUTPUT/evaluation --forti-code-root IMMUTABLE_FORTI_CODE_ROOT/budget_matched_baselines --support-code-root VERIFICATION_SUPPORT_CODE_ROOT --deadline-utc 2026-10-02T13:46:32.725968Z
```

大写路径由调度者填入当前已核验的实际位置，尤其`ARCHIVED_PROJECT`必须与历史`experiment.json`的runtime SHA相同。支持工程须提供`verification_extension`、`reference_replay`等冻结评分工具，Forti路径指向现有已部署不可变包。需要让以上两个代码根目录及本包父目录在`PYTHONPATH`中；队列也向子进程传递这些明确路径。不要将新包覆盖到正在运行的Forti工程里。示例截止时间是当前预算的既定上限，过期后代码返回75，不会自动延期。

三种子训练完成后一次重放269个manuscript测试条件，每条件仍512帧/生成batch8/link8，三模型读取同批输入。输出保留15列现有核心结果、3列新的固定40轮Forti结果和3列新WiMAE结果，共7个方法/21条路线。历史列不重跑推理；复用的Forti外部文件必须与其已验证ZIP内部SHA逐一一致。新模型使用独立`WiMAESliceCE`，不会被旧`wimae_ce`注册入口误装载。

最终统一输出NMSE、BER、噪声方差敏感性、三种子95%区间、共享生成簇配对bootstrap、7方法表格及NMSE/BER图。`verify_wimae.py`同时支持未打包目录和ZIP，只读检查源码、源评分SHA、冻结帧SHA、阶段与训练账本、原始checkpoint元信息、统计区间与ZIP CRC，不导入Torch或运行模型。打包仅保存静态快照，包含新训练状态/私有mask RNG证据、最小官方MIT源码和新的Forti训练交付证据。

```bash
python -m dl_baseline_expansion.verify_wimae --archive WIMAE_EVALUATION_OUTPUT/wimae_results_for_paper.zip --reference-results REFERENCE_RESULTS --forti-code-root IMMUTABLE_FORTI_CODE_ROOT/budget_matched_baselines
```

时间不足会保留已经提交的完整epoch和测试cell，返回75及incomplete；三种子未全部完成时不评分，不完整测试时不生成论文headline表。已完成任务恢复前重新验证账本和输出SHA后跳过，完成marker保留初次字节，避免修改已评分实验身份。正式真实波形路径必须由服务器第一次实际训练记录观察，而不能以合成探针成功代替。
