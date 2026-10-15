# 当前论文的复现矩阵

权威稿为 `technical_fixes_20261014_ack_gpt_claude`（正文15页、补充11页）。原始稿件快照为2份PDF及82个源码/资产文件，本公开代码候选不包含paper_reference；以下稿件资产名仅保留原归档定位，原CURRENT_SOURCE_MANIFEST、README_BUILD、SUBMISSION_CHECKLIST三份包分发元数据不作公开快照输入。原ACK为“OpenAI GPT assisted with code development, and Anthropic Claude assisted with manuscript polishing.”。

这里区分三件事：**A 配方/生成器源码可用**，可以在未来合适环境生成数据另行训练，但不等于原历史张量逐字节复跑；**B 现成证据可核验**，已有分数、CSV、日志和选定权重可按身份/选择条件检查；**C 精确重放仍有缺口**，历史张量、收到的波形、validation重建凭证、epoch proofs或具体图导出器未齐。没有在本次整理中训练、运行producer或重算原测试网格。

## 建议先做的检查

```text
python -B code/scripts/diagnose_gate.py --verify-saved
python -B code/scripts/data_fraction.py verify
```

第一项使用stdlib核验原PT报告、384个model–observation pairs及当前Table5的pooled-linear舍入。第二项核验补齐的12pct配置/12权重及原engine语义，不反序列化权重、不训练。需要新gate推理时，`diagnose_gate.py --output ../gate_full_new`先显示计划，只有明确加`--execute`才运行；`--smoke`只覆盖一权重/一批，不能称全Table5复现。

## 全部显示项

共36项：正文6图/11表，补充3图/16表；31个外部资产与5个inline表均覆盖。表中的“行匹配”指保存CSV/JSON到TeX显示格式检查，不指重新推理或字节相同的图重建。

| 显示项（当前PDF页） | 源文件 | 保存证据/入口 | 边界 |
|---|---|---|---|
| 正文 Fig. 1（p4） | `DOPNet_tmlcn_02_dual_view_overview.pdf`（原稿件归档，公开代码未附） | SVG/drawing source | 静态架构图，不是额外实验 |
| 正文 Fig. 2（p4） | `DOPNet_tmlcn_01_PWB_mechanism.pdf`（原稿件归档，公开代码未附） | SVG/drawing source | 静态架构图，不是额外实验 |
| 正文 Fig. 3（p6） | `DOPNet_tmlcn_04_fusion_reconstruction.pdf`（原稿件归档，公开代码未附） | SVG/drawing source | 静态架构图，不是额外实验 |
| 正文 Fig. 4（p10） | `fig09_support_doppler_snr.pdf`（原稿件归档，公开代码未附） | support_pooled_dynamic_doppler_gap, support_joint_doppler_snr_gap | 具体点/图导出或字节重建未核实（见JSON） |
| 正文 Fig. 5（p11） | `fig06_equal_frame_budget.pdf`（原稿件归档，公开代码未附） | equal_frame_figure6_saved_points | 保存证据可定位；未重生成PDF |
| 正文 Fig. 6（p13） | `fig03_receiver_ber_doppler.pdf`（原稿件归档，公开代码未附） | original_receiver_doppler_figure | 具体点/图导出或字节重建未核实（见JSON） |
| 正文 Table 1（p7） | `main.tex`（原稿件归档，公开代码未附） | common_waveform_training_geometry | 声明/配置，不是测量输出 |
| 正文 Table 2（p9） | `table_training_routes.tex`（原稿件归档，公开代码未附） | common_waveform_training_geometry, added_method_training_recipe_declarations | 声明/配置，不是测量输出 |
| 正文 Table 3（p9） | `table_evaluation_populations_main.tex`（原稿件归档，公开代码未附） | 人口/配置声明 | 声明/配置，不是测量输出 |
| 正文 Table 4（p10） | `table_mechanism.tex`（原稿件归档，公开代码未附） | original_mechanism_paired_gaps | 保存行与显示已核验；不等于运行producer |
| 正文 Table 5（p10） | `table_fusion_diag.tex`（原稿件归档，公开代码未附） | original_pytorch_coefficient_gate | 原PT6模型×8批；正式saved核验/限定新推理 |
| 正文 Table 6（p11） | `table_equal_budget_final.tex`（原稿件归档，公开代码未附） | equal_frame_nmse_summary, equal_frame_ber_summary | 保存行与显示已核验；不等于运行producer |
| 正文 Table 7（p12） | `table_joint_bem_final.tex`（原稿件归档，公开代码未附） | calibrated_core_nmse_summary, calibrated_core_common_ber_summary | 保存行与显示已核验；不等于运行producer |
| 正文 Table 8（p13） | `table_receiver.tex`（原稿件归档，公开代码未附） | original_receiver_diagonal_and_native | 保存行与显示已核验；不等于运行producer |
| 正文 Table 9（p13） | `table_robustness.tex`（原稿件归档，公开代码未附） | original_robustness_pooled | 保存行与显示已核验；不等于运行producer |
| 正文 Table 10（p14） | `table_impairments.tex`（原稿件归档，公开代码未附） | receiver_impairment_saved_nmse | 保存行与显示已核验；不等于运行producer |
| 正文 Table 11（p14） | `table_latency.tex`（原稿件归档，公开代码未附） | historical_gpu_processing_times | 保存行与显示已核验；不等于运行producer |
| 补充 Fig. S-1（p3） | `DOPNet_tmlcn_03_residual_axial_attention.pdf`（原稿件归档，公开代码未附） | SVG/drawing source | 静态架构图，不是额外实验 |
| 补充 Fig. S-2（p8） | `fig08_support_tradeoff.pdf`（原稿件归档，公开代码未附） | support_pooled_dynamic_doppler_gap, support_offsupport_received_energy | 保存证据可定位；未重生成PDF |
| 补充 Fig. S-3（p9） | `fig04_nmse_snr.pdf`（原稿件归档，公开代码未附） | nmse_snr_figure4 | 具体点/图导出或字节重建未核实（见JSON） |
| 补充 Table S-1（p3） | `supplement_body.tex`（原稿件归档，公开代码未附） | added_method_training_recipe_declarations | 声明/配置，不是测量输出 |
| 补充 Table S-2（p4） | `supplement_body.tex`（原稿件归档，公开代码未附） | conditional_interval_definitions | 声明/配置，不是测量输出 |
| 补充 Table S-3（p4） | `table_equal_budget_nmse_detail.tex`（原稿件归档，公开代码未附） | equal_frame_nmse_summary | 保存行与显示已核验；不等于运行producer |
| 补充 Table S-4（p5） | `table_equal_budget_ber_detail.tex`（原稿件归档，公开代码未附） | equal_frame_ber_summary | 保存行与显示已核验；不等于运行producer |
| 补充 Table S-5（p5） | `table_equal_budget_conditional_ci.tex`（原稿件归档，公开代码未附） | equal_frame_nmse_summary | 保存行与显示已核验；不等于运行producer |
| 补充 Table S-6（p5） | `table_comparison.tex`（原稿件归档，公开代码未附） | original_comparison_overview | 保存行与显示已核验；不等于运行producer |
| 补充 Table S-7（p5） | `table_complete_accuracy.tex`（原稿件归档，公开代码未附） | original_complete_accuracy | 保存行与显示已核验；不等于运行producer |
| 补充 Table S-8（p6） | `table_all_modulations.tex`（原稿件归档，公开代码未附） | original_all_modulations_common_ber | 保存行与显示已核验；不等于运行producer |
| 补充 Table S-9（p6） | `table_supp_pilot_cp_detail.tex`（原稿件归档，公开代码未附） | original_pilot_cp_saved_groups | 保存行与显示已核验；不等于运行producer |
| 补充 Table S-10（p6） | `table_supp_sensitivity_detail.tex`（原稿件归档，公开代码未附） | original_sensitivity_saved_groups | 保存行与显示已核验；不等于运行producer |
| 补充 Table S-11（p7） | `supplement_support.tex`（原稿件归档，公开代码未附） | saved_support_operator_geometry | 保存证据可定位；未重生成PDF |
| 补充 Table S-12（p7） | `supplement_support.tex`（原稿件归档，公开代码未附） | saved_support_energy_statistics | 保存证据可定位；未重生成PDF |
| 补充 Table S-13（p8） | `table_support_joint_gain.tex`（原稿件归档，公开代码未附） | support_joint_doppler_snr_gap | 保存行与显示已核验；不等于运行producer |
| 补充 Table S-14（p9） | `corrected_nominal_nmse.tex`（原稿件归档，公开代码未附） | calibrated_nominal_summary | 保存行与显示已核验；不等于运行producer |
| 补充 Table S-15（p10） | `corrected_core_nmse.tex`（原稿件归档，公开代码未附） | calibrated_core_nmse_summary | 保存行与显示已核验；不等于运行producer |
| 补充 Table S-16（p10） | `corrected_core_common_ber.tex`（原稿件归档，公开代码未附） | calibrated_core_common_ber_summary | 保存行与显示已核验；不等于运行producer |

各表完整TeX行、保存CSV选择条件、被选原始行/列、显示匹配记录和当前31资产SHA均在 [REPRODUCTION_MATRIX.json](REPRODUCTION_MATRIX.json)。补充Table S-8虽无label，仍按实际PDF第6页完整纳入。

## 训练与权重边界

原 `config/TMLCN_RECIPES.json` 仍是14配方×3seed=42组合；42个既有selected states保留。新增四个数据比例变体×三seed的12份原权重与12份配置，按实际SHA从既有E档案有界补齐，配置单列 `config/training/data_fraction/`，权重保持 `data/weights/selected/` 原名。因此selected states共54份，不能把原42活动训练列表悄悄说成54个可直接执行路由。

原engine的 `unique_epoch_samples=3000/6000` 对应25%/50%，重复到共享的12000 samples，每batch32、每epoch375次更新；新 `data_fraction.py` 仅list/verify/plan，不伪造38-row reference中的pct全网格评估器。历史大训练张量不在C不妨碍科学生成器用于另一次未来训练，但**exact历史复跑**仍要真实validation attestation、原epoch hashes/proofs及环境门槛，不能用新生成数据或已保存权重替代这些证明。

## 统计、源数据与范围

- original receiver 55-cell/20dB的7.232% diagonal、6.805% native，与33-cell high-mobility的11.828%/11.125%不是同一population。Table8的ICI-aware列用original_retained的`study=receiver, metric=ber_native`，不能误选equal-frame结果。
- nominal236、equal-frame269、classical99/66/33的保存选择条件逐项保留。训练帧预算相同不等于监督、optimizer updates或compute相同。
- Table10来自冻结三模型/五cases的现成NMSE。2000次完整八帧批次paired bootstrap固定模型/cases，不是seed-t；CFO/timing各有自身effective target。CSV内原有BLER列不构成当前新增BLER结论。
- Table11是既有RTX4090、GPU-resident occupied-grid-to-bits、批量摊销处理记录，排除RF/acquisition和pilot lookahead；没有在本机重测或证明单帧实时deadline。
- Table5现由原PyTorch6权重×8批重算，pooled-linear NMSE与IV-D一致。报告的`table5_display`还保留旧mean-of-seed-dB口径，当前应读`aggregates.dopnet.*.pooled_linear_all_seeds`。32帧/64个不同用户观察被六模型共享，384是model–observation pairs。beta=1保留offset并可能改变下游分布，不证明唯一ADD因果贡献。
- 图5的128保存source_row精确匹配现CSV；图4的20点表可核验，具体PDF字节未重建；补图S-3的精确保存点/专属导出器仍UNVERIFIED。四架构图只绑定SVG/绘图源码与模型实现。
- NumPy原脚本、旧日志、FAIL及精确top-k tie反证保存在外部development archive，不能当当前Table5权威或正式quickroute，也不删除反证。

## 最小数据整理

保持已有 `reproduction_inputs/`、`base_scores/`、`base_analysis/`、`classical_verification/`、训练历史和记录图点，避免重复大型validation/原始banks。把五个原字节CSV统一放 `results/paper_evidence/source_tables/`；把原PT384报告放 `results/gate_diagnostics/original_pytorch_20261014/`；相关12pct权重/12配置按有界计划补入。无需复制全旧53-route容器、巨大E banks或无关TWC编码/BLER链路。历史来源中机器绝对路径只是provenance，正式入口依赖相对code/data根。所有移动/复制由根代理最终校验并保留可恢复历史；本矩阵仅写workspace。
