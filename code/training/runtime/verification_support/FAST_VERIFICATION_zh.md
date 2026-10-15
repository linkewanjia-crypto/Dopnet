Historical documentation with machine paths redacted. Original bytes remain in the scientific archive.
This document is not a valid signed execution plan; use the portable entry and a fresh prepare.

# 经过数值检查的验证提速

当前入口为 `run_fast_verification.py`。上传并解压增量包到 `__HISTORICAL_MACHINE_PATH_REDACTED__` 后执行：

```bash
cd __HISTORICAL_MACHINE_PATH_REDACTED__ && python run_fast_verification.py --reference-results __HISTORICAL_MACHINE_PATH_REDACTED__ --output __HISTORICAL_MACHINE_PATH_REDACTED__ --suite manuscript
```

先复用或生成独立校准观测，然后在这些观测上检查原实现与缓存实现的设计矩阵和五个 ridge 的系数、残差信息是否逐位一致。全部检查通过后才应用优化。原有神经模型评分直接复用，不训练模型。模型接收链时延仍仅在 GPU 测量。

提速内容：

- 同一公开导频布局的 BEM 设计块保存在有容量上限的主机缓存中。键包含该符号两个用户全部导频的实际字节，不假定导频固定，也不缓存接收信号、真实信道或预测结果。缓存块由原函数产生，不重新计算近似设计。
- 独立校准中，一帧的五个 ridge 候选复用 Gram 矩阵，各候选仍独立执行原 complex128 Cholesky、精化和原始正规方程残差检查。普通 Tensor 用版本计数检查变更；inference Tensor 用私有快照检查。
- 原 full-band ICI 接收机的符号分块在独立校准数据上比较 4、8 和完整符号数，仅采用符号输出、BER 比特计数和残差检查均通过的分块。

原生成器、测试 seed、每批 8 帧、每条件 512 帧、链路分块、候选网格和误差阈值保持原协议。测试数据仍完整通过原 SHA-256 波形检查后才保存。缓存放在主机，不占用原 3GPP 生成所需的大块显存。

`execution_optimization.json` 保存服务器、源码哈希、一致性检查、分块选择和分段计时。`execution_*.json` 保存实际缓存命中与阶段耗时。这些文件自动进入最终回传包。显示的重复帧计算加速倍数只适用于相应工作负载，不能直接当作全套结果生成的加速倍数。

保留原校准绑定的文件字节，因此完整的旧校准可复用。更换服务器后重新测量传统求解后端。断点恢复会校验并复用已完成的条件。

最终下载：`__HISTORICAL_MACHINE_PATH_REDACTED__`。

旧评分中仅评估 NMSE 的条件没有 `bits` 字段。兼容适配器仅在该条件声明计算 BER 时读取原 `bits`；NMSE-only 条件不读取它，不补造 BER 分母。兼容适配器的哈希与每个新评分一起记录，原校准绑定文件保持原字节。
