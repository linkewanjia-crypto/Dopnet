# FortiTran：与 DOP-Net 相同的数据训练预算

本目录只实现 FortiTran 的 42、43、44 三种子实验。每个种子训练恰好 40 个 epoch，每个 epoch 使用 12,000 个新生成帧；保持 Nr=8、Nf=128、T=25、Nu=2。验证使用原有 2,000 帧、固定 SHA-256 的缓存，最终比较使用原有 269 个论文测试条件。其他深度基线由独立实验计划处理。

采用源码指导的 Adam/MSE、有效 batch 512、学习率 0.001 和每 epoch 0.995 衰减、FP32。40 个 epoch 不提前停止；原论文的 1,000 epoch 上限和作者仓库的 patience=3 作为原始来源说明保留。这里的名称和声明为 `fortitran_budget_matched`、`source_guided_equal_budget`，不声明原始 SISO 数据集的精确复现。

服务器运行顺序：

1. `python -m budget_matched_baselines.prepare`：使用 `--project`、`--reference-results`、`--validation-cache`、`--output` 准备独立计划。
2. 如果继续现有训练前缀，先确认旧 worker 停在已完整提交的 epoch；运行 `python -m budget_matched_baselines.migrate_prefix --project PROJECT --parent-manifest OLD/recipe_manifest.json --manifest NEW/recipe_manifest.json --parent-implementation OLD_CODE/author_baseline_reproduction --seeds 42`。
3. `python -m budget_matched_baselines.start`：传入上述三个路径，以及 `--verification-code-root`、`--verification-output`、`--output NEW`、明确授权的 `--deadline-utc` 和 `--skip-resource-probe`。启动器串行训练三种子，再评分和生成图表。

迁移保存旧 manifest、配置、代码、checkpoint 和 history，逐项核验模型、optimizer、scheduler、RNG 与历史状态，不覆盖旧实验。初始迁移快照与后续完整 40-epoch 结果分别保留证据。截止时间到达只产生可续训的未完成状态，不算完成实验。

最终文件为 `NEW/evaluation/budget_matched_results.zip` 及其 SHA-256 文件。`verify_author_delivery` 负责只读交付核验。

本机只做 AST、标量与协议元数据检查，不训练、推理、生成波形或计算科学指标：

```bash
python -B -m unittest budget_matched_baselines.test_recipe budget_matched_baselines.test_metadata_protocol -v
```
