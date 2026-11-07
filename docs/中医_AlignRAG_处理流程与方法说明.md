# 中医 AlignRAG 处理流程与方法说明

> **历史文档：回答感知 Critic。** 下文记录旧版 `(q,D(q),y_weak)` 监督与读取初答的流程，不是当前 EviCR 协议。当前离线/在线 Critic 均仅输入 `q,D(q)`，请以仓库 README、`DATA_CONTRACTS.md` 和 `run_evicr.py` 为准。本文件保留用于来源追溯。

## 1. 目标与总体框架

本流程在冻结的 R4 检索体系上构建面向中医问答的证据一致性批评器（Critic），用于减少回答中超出证据范围的诊断、处方、剂量、疗效和处置性表述。方法由两部分组成：离线阶段利用独立强教师构造批评监督并训练轻量 Critic；在线阶段对初始 RAG 回答进行一次证据约束下的批评和精修。该流程服务于研究性文本问答，不用于临床诊疗。

```text
Build 查询 q
    |
    v
冻结的 R4 混合检索 --> Top-3 证据 D(q)
    |                         |
    |                         +--> 弱模型：q, D(q) -> y_weak
    |                         +--> 独立强教师：q, D(q) -> y_strong
    |                         +--> 批评教师：q, D(q), y_weak, y_strong -> c*
    v
无泄漏 SFT 数据：q, D(q), y_weak -> c*
    |
    v
QLoRA 微调得到 Critic C_theta

在线：q, D(q) -> y0 -> C_theta(q, D(q), y0) = c0 -> y1
```

其中，`y_weak` 为弱模型初始回答，`y_strong` 为不读取弱回答的独立强教师回答，`c*` 为强教师给出的结构化批评标签；`y0`、`c0`、`y1` 分别表示在线初始回答、Critic 批评和一次精修后的最终回答。流程实现位于 `prepare_critic_seed_set.py`、`generate_critic_supervision.py`、`export_critic_sft.py`、`train_critic_lora.py` 和 `run_tcm_alignrag.py`。

## 2. 记号与数据边界

设 Build 划分中的应用样本集合为 $\mathcal{A}_{B}$，Build 边集合为 $\mathcal{E}_{B}$。每条边 $e \in \mathcal{E}_{B}$ 将一个历史应用样本与一条知识项关联，构成检索文档：

$$
d_e=(q_e,a_e,k_e,r_e),
$$

其中 $q_e$ 是历史问题，$a_e$ 是该历史问题的回答，$k_e$ 是关联的 K 知识文本，$r_e$ 是关系类型。当前实现使用三类关系：`diagnostic_support`、`treatment_or_operation_support` 和 `explanatory_support`。

对目标问题 $q$，检索结果写作 $D(q)=\{d_{(1)},d_{(2)},d_{(3)}\}$。每个证据块按如下可追溯格式输入模型：

```text
[E_j | e_id | relation]
历史问题：...
历史回答：...
K 知识：...
```

这里的历史回答仅可作为相似案例参考，不能被表述为当前提问对象的既定事实；K 知识可作为可复用依据。需要特别区分两类回答：目标样本原始 `answer` 从不进入种子或 Critic 的学生输入；而检索到的其他 Build 样本的历史回答 $a_e$ 是 R4 证据字段的一部分，受“仅作案例参考”的提示约束。

## 3. Build 种子集构建

训练种子由 `prepare_critic_seed_set.py` 从冻结的应用数据和 Build 边数据中选择，输出 `data/critic_seed_3000.jsonl` 及同名 manifest。具体规则如下。

1. 仅保留 `release.split == build` 的应用样本，且其规范化后问题长度不少于 8 个字符、至少关联一条 Build 边。
2. 以删除空白符后的问题文本去重；同一问题的多个原子片段只保留 `application_id` 字典序最小的代表，避免同题重复进入监督数据。
3. 以“来源数据集 + 关系类型组合”为分层单元，采用最大余数法分配各层配额。对层内样本按 $\mathrm{SHA256}(s:\mathrm{application\_id})$ 稳定排序，其中随机种子 $s=20261011$，从而使选择过程可复现。
4. 输出仅保留 `sample_id`、`application_id`、`question`、来源和关系类型等元数据，不写入目标样本的 `answer` 字段。

当前冻结 manifest 中共有 24,069 条合格且问题去重后的 Build 候选，最终抽取 3,000 条：文本来源 2,038 条、视觉来源 962 条；其中包含 2,550 条单一治疗/操作支持关系、158 条诊断+解释关系、143 条单一解释关系、124 条单一诊断关系和 25 条诊断+治疗/操作关系。这些数字描述的是当前种子文件的构成，不代表模型效果。

## 4. R4 混合检索与证据构造

### 4.1 索引

调用 `R4/run_aek_hybrid_r4.py` 时，仅以 Build 边物化全局检索库。词法索引使用 SQLite FTS5；中文文本先切分为相邻二元字片段，英文和数字保留为连续 token。稠密索引使用 BGE-M3，对以下文本拼接后编码并 L2 归一化：历史问题、历史回答和 K 知识。

对当前问题所属的应用标识 $a(q)$，无论词法检索还是稠密检索，均排除满足 $\mathrm{application\_id}(d)=a(q)$ 的文档。该自排除机制避免模型从同一原始应用的边中间接读取自身信息。

### 4.2 候选召回与重排序

词法通道从全局索引召回 BM25 前 100 条；语义通道采用归一化 BGE-M3 向量内积（余弦相似度）召回前 100 条，并额外保留 8 个缓冲位置以补偿自检索剔除。两路候选并集经自排除后进行 min--max 归一化：

$$
\widetilde{s}(d)=
\begin{cases}
1, & \max_{d'\in\mathcal{C}}s(d')=\min_{d'\in\mathcal{C}}s(d'),\\
\dfrac{s(d)-\min_{d'\in\mathcal{C}}s(d')}
{\max_{d'\in\mathcal{C}}s(d')-\min_{d'\in\mathcal{C}}s(d')}, & \text{otherwise},
\end{cases}
$$

其中 $\mathcal{C}$ 为候选集合。令 $J_2(x,y)$ 为去除标点后的二元字符集合 Jaccard 重叠率，$\rho_{r}(q)\in[0,1]$ 为问题中与关系 $r$ 对应的意图词命中数相对最大命中数的归一化值，则最终排序分数为：

$$
\begin{aligned}
S(d\mid q)=&\ 0.48\,\widetilde{s}_{\mathrm{BM25}}(d)
+0.35\,\widetilde{s}_{\mathrm{dense}}(d)\\
&+0.10\,J_2(q,q_d)
+0.07\,J_2(q,a_d)
+0.05\,J_2(q,k_d)
+0.05\,\rho_{r_d}(q).
\end{aligned}
$$

关系意图词覆盖辨证/证型/诊断、治疗/用药/针灸和原因/机制/解释等问题表述。关系项是软加分特征，不会过滤任何候选。按 $S(d\mid q)$ 降序后取 Top-3，得到 $D(q)$；因此该分数是排序函数而非权重和为 1 的概率分布。

## 5. 三阶段教师监督构造

`generate_critic_supervision.py` 将生成拆为三个严格隔离的阶段，所有阶段都仅接受 Build 划分的 R4 检索输出，默认温度为 0，单次最大生成长度为 700 tokens。

### 5.1 初始弱回答

弱模型根据问题和证据生成：

$$
y_{\mathrm{weak}}=G_{\mathrm{weak}}(q,D(q)).
$$

提示词要求仅陈述证据支持的内容；历史问答不可视为当前事实；不得补充无依据的具体诊断、方剂、剂量、疗效或处置；证据不足时必须明确说明。默认每个证据字段截断至 850 个字符。

### 5.2 独立强教师回答

强教师在相同问题与证据条件下独立生成参考回答：

$$
y_{\mathrm{strong}}=G_{\mathrm{teacher}}(q,D(q)).
$$

该阶段的提示词和函数签名均不接收 `y_weak`。独立性是必要条件：若强教师预先读取弱回答，则后续差异可能反映锚定效应，而非对同一证据的独立判断。

### 5.3 教师批评标签

批评教师同时读取 $q$、$D(q)$、$y_{\mathrm{weak}}$ 和 $y_{\mathrm{strong}}$，生成结构化标签：

$$
c^*=T(q,D(q),y_{\mathrm{weak}},y_{\mathrm{strong}}).
$$

`c*` 必须是一个 JSON 对象，包含 `verdict`、`issues`、`revision_instructions` 和 `evidence_boundary`。`verdict` 只能取 `needs_revision`、`acceptable` 或 `insufficient_evidence`；每个问题项包含位置/断言、原因、可引用的证据编号和可执行修订建议。允许的错误类型为：

- `unsupported_claim`：证据不支持的断言；
- `historical_case_as_current_fact`：将历史案例误写为当前事实；
- `missing_constraint`：遗漏限制条件；
- `overstatement`：过度确定；
- `not_answering_question`：未回应问题；
- `evidence_misuse`：错误使用证据。

批评教师仍以 $D(q)$ 作为唯一事实判断依据，`y_strong` 仅用于帮助发现弱回答的缺陷，不能被自动视为事实。生成脚本保存输入文件 SHA-256、提示词版本和模型来源；发生网络异常时最多重试 3 次，且支持按已完成 `sample_id` 续跑。API 密钥只从环境变量读取，不写入 JSONL、清单、脚本或论文。

## 6. 无泄漏 Critic SFT 数据导出

`export_critic_sft.py` 是教师信号与学生训练之间的边界。其输入必须是完成的 `critique` 阶段 JSONL，导出时逐条执行以下校验：

1. 样本必须为 Build 数据，且不得是 `dry_run` 预览记录；
2. 拒绝任何带目标原始 `answer` 字段的记录；
3. 每条记录必须有 1--3 个规范化 R4 证据块、非空问题和弱回答；
4. 严格解析批评 JSON：`issues` 最多 8 项，`revision_instructions` 为 1--8 项，错误类型、裁决值和证据编号必须合法；若裁决为 `needs_revision`，至少应存在一个问题项；
5. 对 $(q,D(q),y_{\mathrm{weak}})$ 计算指纹并去重。

导出的 Critic 训练样本为三轮消息：

$$
x_i=[\mathrm{system};\ q_i,D(q_i),y_{\mathrm{weak},i}],\qquad
t_i=c_i^*.
$$

也就是说，学生输入只有问题、R4 证据和初始弱回答，明确不含 `y_strong` 及目标样本的原始答案。数据以

$$
\mathrm{SHA256}(\texttt{tcm-alignrag-critic-sft-v1}:\mathrm{application\_id})\bmod 100
$$

确定性地划入训练集或验证集；默认 10% 为验证集。脚本分别输出 `critic_sft_train.jsonl`、`critic_sft_validation.jsonl`、被拒记录及 manifest。远端全流程入口将最少接收数设为 2,500 条，以防教师 JSON 质量不足时静默训练。

## 7. QLoRA Critic 训练

`train_critic_lora.py` 使用导出的三轮消息训练 Qwen 系列 Critic。训练前会再次验证全部记录均为 Build 数据、消息顺序为 `system-user-assistant`，并检查前两轮消息不含 `y_strong` 或“强教师的独立参考回答”等泄漏标识。

令 $z_i$ 为经聊天模板编码后的 token 序列，$\mathcal{I}_i^{A}$ 为 assistant 批评 JSON 的 token 位置集合，则训练目标是条件语言模型损失：

$$
\mathcal{L}(\theta)=-\sum_i\sum_{t\in\mathcal{I}_i^{A}}
\log p_{\theta}(z_{i,t}\mid z_{i,<t}).
$$

问题、证据和弱回答部分的标签全部设为 `-100`，因此不计入损失；只有 Critic 输出 token 参与监督。实现还要求“完整消息 token 序列以前两轮提示 token 序列为前缀”，否则中止训练，以避免聊天模板错位导致错误掩码。

默认训练配置为单张 CUDA GPU 的 4-bit QLoRA：NF4 量化、双重量化、最大长度 4,096、2 个 epoch、学习率 $2\times10^{-4}$、微批大小 1、梯度累积 16、LoRA rank 32、$\alpha=64$、dropout 0.05，并启用梯度检查点。LoRA 注入 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj` 和 `down_proj`。超过最大长度的样本会被剔除并在训练 manifest 中记录。最终保存 PEFT adapter 和 tokenizer，而非合并后的基础模型。

## 8. 在线单轮 AlignRAG 推理

在线推理由 `run_tcm_alignrag.py` 执行，输入为不可变的 R4 检索结果。对于测试问题，先由生成模型产生初始回答，再由已训练 Critic 审查，最后由生成模型据批评进行一次精修：

$$
\begin{aligned}
y_0&=G(q,D(q)),\\
c_0&=C_{\theta}(q,D(q),y_0),\\
y_1&=G(q,D(q),y_0,c_0).
\end{aligned}
$$

Critic 不会在此阶段自行检索，也不可见目标原始答案或教师强回答；精修提示词再次要求只保留显示证据支持的内容，并不得提及模型、批评或编辑过程。当前实现固定为 1 轮，输出记录包含 `initial_answer`、`critic`、最终 `answer`、格式化证据和 `alignrag_config`。默认期望输入划分为 `test`，如果遇到其他划分会直接报错，从而支持冻结 Test-111 上的成对 RAG+ 与 RAG+AlignRAG 评测。

## 9. 数据隔离、复现与质量控制

整个流程通过以下机制防止数据泄漏并保留可复现性：

- 训练种子、教师监督和 LoRA 训练均限定为 Build；Test 111 不参与这些步骤；
- R4 索引仅由 Build 边建立，并对每个查询剔除同 `application_id` 的证据；
- 目标样本原始答案不进入种子和 Critic 学生数据；`y_strong` 在 SFT 导出边界被移除；
- 种子选择、训练/验证划分和源文件 SHA-256 均可追溯；输出文件拒绝无提示覆盖；
- `tests/test_contracts.py` 检查 3,000 条种子契约、教师提示独立性，以及 SFT 导出和训练前的泄漏约束。

在报告实验结果时，应固定 R4 检索配置、生成模型、Critic adapter、提示词版本及测试集；仅在同一冻结问题和同一 Top-3 证据条件下，对基线 RAG+ 与加入单轮 Critic 的 RAG+AlignRAG 进行配对比较。上述代码提供流程与控制措施，不应被表述为未经实际运行验证的性能提升。

## 10. 可直接用于论文“方法”部分的表述

为提升中医检索增强问答的证据一致性，本研究在冻结的 R4 混合检索框架上构建 AlignRAG。首先，仅从 Build 划分中按来源和关系类型分层抽取 3,000 个去重问题，并以全局 BM25 与 BGE-M3 稠密检索的候选并集为基础，结合问题与历史问题、历史回答、K 知识的二元字符重叠及关系意图软加分进行重排序，保留 Top-3 证据；检索时严格排除与当前问题同源的应用样本。随后，弱模型和独立强教师分别在相同证据条件下生成初始回答与参考回答，批评教师基于问题、证据、两类回答生成包含裁决、问题类型、证据编号、修订建议和证据边界的结构化批评标签。为避免特权信息泄漏，Critic 的监督输入仅包含问题、Top-3 证据和弱回答，不含强教师回答或目标样本原始答案；在此基础上以仅对 assistant 批评 token 计算损失的 QLoRA 进行监督微调。推理阶段固定执行一次“初始生成--证据批评--回答精修”链路，Critic 不执行额外检索且仅依据同一组冻结证据提出修订约束，从而在不改变检索结果的前提下约束最终回答的证据边界。
