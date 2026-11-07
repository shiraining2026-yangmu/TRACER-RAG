# Qwen3.5 历史完整两轮链路的三模型复现

> **历史运行记录。** 本页与其链接中的“当前”“正式”等指当时的历史实验，不是当前公开的证据前置 Critic 协议。当前离线和在线 Critic 只接收 q、D(q)，全量推理入口为 `run_evicr.py`。

> 2026-10-20 后续指令已将当前任务改为 TER 错题证据修复，本两轮版本停止并保留。当前执行入口和唯一正式目录见 [TER_WRONG_ONLY_THREE_MODEL.md](TER_WRONG_ONLY_THREE_MODEL.md)。以下为历史配置记录。

2026-10-20 用户明确选择完整历史链路。模型依次为 Mistral-7B-Instruct-v0.3、Gemma-2-9B-it、Yi-1.5-9B-Chat。

流程：冻结证据 → 当前候选生成 TER 原始初稿 y0 → 固定 Qwen3.5+LoRA 回答 Critic → 当前候选精修为 y1 → 离线选择 y1 错题 → 固定 Qwen3.5+LoRA v3 证据 Critic → 当前候选作答 → 与 y1 正确题合并。

历史代码中第二轮生成器及 change_support 门控使用的 initial_answer 仍为原始 y0，而不是 y1。新执行器保留这一行为；答案隔离检查确保初稿属于当前候选，没有使用历史 Qwen 回答。

只报告普通向量 RAG、TER 直接生成、完整 TER+EviCR v3 三组。y1 中间组保留用于审计，不作为第四组加入最终表格。

## 冻结设置

- 回答 32 token，回答 Critic 384 token，证据 Critic 640 token；各 Critic 格式修复最多一次。
- 证据字段上限 850 字符，Top-3 冻结检索；greedy，NF4 4-bit double quantization。
- 为复刻历史规则，无法提取选项且格式修复失败时默认 A。强制默认项在逐题输出及汇总中显式标记，不视作模型直接合规输出。
- 2021/2022 原始题数 566/596，可评分 554/584，共 1,138。标准答案只在离线评分/错题路由时读取，不进入模型提示。
- 模型使用其官方模板；Gemma 保留必要的 system 合并及 eager attention 适配；用户在 2026-10-20 明确要求普通生成和两轮精修固定采用 4 题批量。每个模型启动后执行共享显存的 32/384/640-token 预检，不静默回退到 2 题。
- 已知题库病例错位、检索及训练同题重叠沿用历史版本；这些结果属于依赖离线标签的历史条件性修复诊断，不是修复数据污染后的盲态部署实验。

## 版本与恢复

当前版本：`qwen35-two-pass-batch4-v2`。

当前唯一结果目录：`/home/ubuntu/tcm_rag/alignrag_tcm/questionbank_eval/model_benchmarks/qwen35_replay_batch4_v2/`。

原 `qwen35_replay_v1` 的 2 题批量精修及所有中间结果原位保留。Mistral 已完成的四份普通向量 RAG、TER 结果仅在校验完整 ID、冻结证据、原始生成代码哈希、32-token 上限、4 题批量与官方模板运行环境后按字节复制到新版本，并记录 `baseline_reuse.json`。2 题精修不会复制，新版本从同一候选 TER 原始 y0 重新执行两轮。

原 `adapted_v2` 使用 128 token，且缺少回答 Critic 阶段，其结果原位保留，不混入新协议。下载完成的模型权重和冻结证据复用；新协议的每个阶段按逐题原子提交结果断点续跑。

入口：`run_three_model_qwen35_replay.py`。服务：`multimodel-exam-benchmark.service`，新覆盖配置 `40-qwen35-batch4.conf`。服务保留开机自启、用户 linger 和失败 90 秒重启；模型阶段错误先重试 3 次，始终先恢复当前模型，再下载下一个模型。TERM 时将当前阶段记为 interrupted。

每题回答上限 32 token、回答 Critic 上限 384 token、证据 Critic 上限 640 token，不改变提示词、格式修复次数、评分或错题路由规则。批量 4 与批量 2 的浮点计算可能影响中间文本，因此新版本不将两者的精修预测混合。2026-10-20 的单独探测显示 Mistral 4 题第一轮吞吐约为 2 题的 1.30 倍，证据 Critic 吞吐约为 1.81 倍；这不是三个模型的整卷速度保证。

完成条件：独立重算三组及中间 y1 的八份分年成绩，检查 1,162 题全部覆盖、冻结证据、原始 y0、v3 的 y1 错题集合与正确题保留；生成 summary 后才清理对应候选权重。清理不删除原始结果，权重可按 model_source.json 重新下载。

最后生成 `all_models_summary.json`；每个模型保留 manifest、模型版本/校验和、两份部署预检、逐阶段 generation_contract、两轮 Critic 和逐题评分。
