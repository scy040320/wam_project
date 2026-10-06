# 整体恢复训练合同

本合同是新版本的输入、监督、控制组和整体训练准入约束。它没有拟合模型，不改变旧冻结模型、旧数据、银标、阈值或原 Gate3。单元测试通过不代表真实数据已可训练，也不代表归因带来了收益。

## 使用前提与恢复目标

- `required_use`：保持候选原计划的使用时刻 `0..15`。关节任务保留冻结的 `typed_gate_effect` 前提，不要求抬起整个抽屉。
- `next_boundary_goal`：查询前冻结、由预测归因与当前 belief 来源证实仍缺失的事实，在下一动作块边界 `16` 验收。不能由实际执行结果选择目标。
- `epistemic_goal`：独立的 `observer_evidence_available@16`。仅在部署侧 before 质量确实不足、归因为遮挡/未决、计划预先声明主要或次要信息获取用途时启用。它不宣称物体物理不可见、冲突已解决或已抓持。

`0` 时刻必须由可信 actual-before 核验，或走安全 fallback。未来恢复概率只能提供有界排序残差，不能写回当前 belief、提前证明事实、推迟使用时刻或越过 entry 门控。secondary 信息目标不改变候选 primary stage 或其硬前提。

## 整体训练准入

七个前提接口是 `target_visible`、`target_pose_current`、`target_reachable`、`grasped`、`lifted`、`receptacle_visible`、`place_ready`；另有独立的 observer 信息目标。`grasped` 的首轮语义可为经独立合同审核的“当前仍携持”，不是力闭合。接触不等于抓持，关节变化不等于新定位，释放不等于 placed。`execution_consistent` 是已发生动作的硬记录，不是可逆恢复头；`placed` 留作终局监督。

全部接口先通过各自语义定义与未知安全合同。定义通过不等于有经验监督：每个权重与目标置信度乘积非零的未来 use/goal，必须同时有 train/val 的可信正、负证书，以及同一完整来源池、同一时刻的不同候选正负对照。历史只记录被选动作的行不能冒充整池反事实。

任一激活目标缺失监督，整体 `training_ready=false`；不准部分训练、不准零头拟合、不把 mask 当负例。所有方法共享同一显式 `admitted_keys`，包括 effect-only；范围外头不读取概率、不计分。

observer 使用独立的 quality 证书：`actual_rgb_measurement_verified`、`no_physical_absence_claim`，以及 hash 绑定的实际 after step16 quality JSON、真实 RGB、relevance maps、冻结观察器代码和权重。消费者按既有 `map_endpoint` 与可靠视角/冲突条件重算质量布尔值，不能只信 passed 字符串。正例仍需要独立 actual RGB 身份证书；该证书与 after 观察器 False 冲突时须保持 masked。负例仅指冻结观察器的质量不足，不表示物体物理不存在。物理谓词继续要求原 `actual_rgb_observable` 证据，不能借 observer False 充当 held/pose 等物理负例。

## 原始输入与来源隔离

新 `X` 必须明确含 actual-before 双 RGB、9D proprio、部署质量，以及各候选预执行预测双终态 RGB、16 个 visual value/mask、16×7 计划动作、语言与实体/关系绑定。不能把旧归因加权 604 维向量改名当作原始 B。

来源检查实际解码 RGB，核验 proprio/动作 NPY 与 inline 数值完全一致，核验 quality/visual/binding JSON 内容完全一致。文件哈希与 role 字符串仍不能证明时间因果性：独立 producer 审计须绑定完整 deployment payload、真实原查询/观测时刻，以及预测归因与 belief 变化来源。

private runtime snapshot 和 query 仅放 `row.before_lineage`，用于快照/池来源审计，不能交给部署编码器。实际执行、物理教师和观察证书只作 `Y`。原 canonical `768` 个 rank 行与 `957` 个 auxiliary 行逐行内容/顺序和源文件哈希保持不变；构造探针不获得虚构终局成功标签。

## 公平控制

full、masked、shuffled、no-DAG、effect-only 共享原始候选 B、目标/使用时刻、模型头准入、reference、backbone、cap、选择包装、tie、硬门控、fallback 和预算。

- masked：恢复需要权重全部置零，但不能删普通候选视觉输入。
- no-DAG：只清传播需要的信用，保留直接来源需要。
- `shuffled_recovery_need_weights`：固定 seed0，在各 split 内按完整 dataset/suite/task/state 来源组作 derangement；只置换相同时刻/类型/谓词的需要总预算，按接收方原 cause 比例分配。无对应目标或只有一组时明确置零。它保留 cause/path/source，不读 Y、不换标签；**不是完整归因器重跑或完整归因删除消融**。
- effect-only：同一个已准入恢复预测器与普通候选输入，使用通用效果权重，不使用归因需要；用来排除普通效果预测收益冒充归因收益。
- frozen backbone：原候选排序控制；另外报告官方 value-only 与事后 oracle 上限。

实际五路控制表、shuffle donor 记录与 producer/control 审计须按 payload digest 精确绑定，不能只填测试名称或 passed flags。

**上述五路表只是直接恢复需要评分通道的控制，不是完整归因推理或硬 belief 依赖传播的消融。** 整体真实数据控制 producer 尚未完成，整体门明确返回 `complete_pipeline_controls_ready=false` 和 `complete_attribution_and_belief_control_producer_not_available`；即使评分通道输入与监督检查都过，也不启动训练或宣称各模块必要性。

另已实现 `attribution_control_factory.py` 的当前块控制准备接口：从同一已核验 prior 和实际 before 证据重新计算 full、无 DAG 和打乱预测通道的 belief；屏蔽版本显式跳过分类/因子更新，避免置信度为零的 UNKNOWN 继续传播。固定 seed0 在 split 内作跨 `(task,state)` 的预测通道双射，不能双射则明确不可用。候选、原始 value、预算、backbone、cap 和使用时刻保持一致，实际门控结果差异完整记录。

该接口的 26 项测试通过，仍不代表真实数据控制审计通过。屏蔽版本保留共用的感知质量与执行记录证据，不等于删除全部 D18 感知模块；各版本保留共同 prior 中既有的历史归因，不等于重放完整历史闭环。后者需各实验臂从其自身历史重新更新，并报告真实执行与 fallback 成本。不能将当前块控制或需要权重控制的结果冒称为整个归因模块/依赖图的最终必要性证据。

## 入口

`audit_whole_training(bundle, base_path=...)` 只审计；`score_pool(..., admitted_keys=...)` 只作评分分解，不拟合、不认证真实收益。生产必须显式传同一个 whole-gate `active_heads` 范围，无参数默认仅供合成接口测试。

独立 CLI：

```text
python scripts/audit_joint_recovery_training.py --input NEW_BUNDLE.json --output NEW_AUDIT.json --library-root NEW_CODE/lib --expected-module-sha256 FROZEN_SHA
```

入口核验 scoped 模块 origin 与 SHA，拒绝覆盖已有审计输出。真实整体门未通过时不得启动拟合；接口测试、技术审计与实证方法有效性分别报告。
