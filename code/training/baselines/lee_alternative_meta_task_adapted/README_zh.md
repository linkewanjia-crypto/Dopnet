# Lee A-ML同预算训练集成资源门控补丁0.3.1

此版本依据用户提供的Lee全文及已冻结0.1.0核心，保留0.2.0已审查训练机制，并新增串行资源探测/三seed训练控制器、独立配置、CUDA训练器、分块完整二阶更新、整轮状态提交/恢复、受限元信息检查及串行硬件探测。它尚未部署，尚未取得实际CUDA二阶资格，尚未训练。18项本机检查只使用标准库、AST、元信息、无效占位文件和小标量解析式；没有导入Torch或NumPy、生成真实数据、运行模型或测量GPU。

论文来源及适配依据见原`baseline_reproduction_sources/LEE_META_40E_SOURCE_CONTRACT_20261002.md`。原算法的并联SFT+ADD、ADD单内步完整二阶离线元学习和在线联合训练保留。1元学习epoch+39在线epoch、batch32内16/16任务划分、K1、alpha/beta=1e-3、d56/s12/m4、公共历史验证选优、原128/8/25/2/1024几何均按来源明确声明；不能称作者完整参数/数据集复现，也不能把375个元更新称为作者的1次离线iteration。

每个seed的40轮全部使用原全局生成epoch1..40，每轮12000 fresh接收帧、375个更新，验证只读原2000帧cache。元学习仅更新ADD；在线完整25符号复数MSE联合更新两分支，Adam常学习率1e-3，FP32，无scheduler/early stop/clip。验证的公共data CSI val_mse尺度与其他方法一致，只在在线epoch2..40按最早最小值选优。模型输入只有LS两通道；元任务划分只读训练批的fd元信息，测试不作梯度更新。

`commits/epochNN`目录内完整state.pt和签名commit.json构成唯一提交单位。完整模型、Adam/RNG、配置/源码/cache绑定、全账本和生成epoch身份都保存；未发布的staging目录不计轮数。恢复先核验所有连续提交SHA，再恢复模型/优化器，最后恢复Python/NumPy/Torch/CUDA RNG。生成器按冻结源码的seed+globalepoch确定数据，不重新从阶段epoch1开始。阶段2只初始化一次联合Adam，不重新seed。RNG存在核验不冒称逐位连续证明。

`probe.py`必须在原/Forti/WiMAE完整checker通过、所有controller/worker自然退出后串行运行。它只用真实CUDA合成shape完成16support+16query的二阶更新及32帧联合梯度更新，不生成真实波形，不能证明真实全path吞吐或精确显存峰值。非重入重算函数显式接收theta/phi参数，防止functional_call结束后重算错用base参数。Torch上限6GiB，原生成32/link8不减。实际worker首整轮的Torch分配器峰值单独记录，仍须补充实际全path资源观察，不能追溯假PASS。

服务器真实训练必须绑定passed的同源实际探测receipt和原/Forti/WiMAE完整交付行政receipt。所有CLI硬编码原UTC2026-10-02T13:46:32.725968截止，无延长参数。到截止保留已提交整轮、标incomplete/exit75；部分轮丢弃资格，模型字节不删除。

此候选没有Lee新增269冻结评分、八方法24routes报告、CI/资产/科学结果ZIP checker、Lee实际时延或完整队列控制器。现有WiMAE七方法21routes与F/W72时延源完全冻结；不能热改扩充为Lee。后续评分、报告、交付checker和Lee时延采用独立新版本接入此冻结训练来源。鉴于原截止不变，本版本经完整源/元信息审查后可先部署并串行资源检查及训练；后续流程尚未完成必须明确保留incomplete，不能以训练完成或发布包替代整体验收；本候选prepared、18 tests通过或单seed训练metadata PASS均不是实验完成。原21:46截止、30分钟检查和最终统一本地核验后退还条件不变。

没有新增独立agent审查。作者code未取得；合法全文缺口已解除。


0.3.0仅增加串行控制：所有原/F/W交付与新F/W72时延须实际已验收且全部worker退出；probe候选1/2/4/8逐个真实执行，按两阶段合计吞吐选择6GiB内最快passed候选，选后绑定该manifest和probe receipt依次训练42/43/44。每seed自然40后一次受限元信息检查；控制器训练阶段全通过仍标scoring_pending/all_experiments_complete=false。没有自动延长或测试标签适配。


0.3.0首次真实原3GPP生成因全进程6GiB allocator门控OOM，0个完整epoch提交，失败queue/日志/四个passed合成probe完整保留。0.3.1不热改0.3.0：硬件probe仍全进程6GiB，真实训练总进程可用原GPU容量减3GiB，每次估计器更新相对真实波形驻留基线新增分配不得超过6GiB。记录完整波形峰和估计器增量峰，生成32/link8、模型/完整二阶算法/epoch/优化器/数据身份/截止均不变。原四passed probe不移植为新来源资格；新版本重新串行实际资格。未生成首批不当fullpath通过，training_only仍不代表完整实验。
