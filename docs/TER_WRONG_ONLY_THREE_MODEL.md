# 三模型 TER 错题证据修复

> **历史运行记录。** 下文“当前”等时间表述均属于当时的记录；答案键选错题属于条件性修复诊断，不是当前 q、D(q) 全量盲测 EviCR 的正式结果。当前公开入口为 `run_evicr.py`。文中 v4/v5 部署分支不在本次精简发布范围。

> 2026-10-21 当前：Mistral 已完成，三组分别329/1138（28.91%）、384/1138（33.74%）、521/1138（45.78%），以原 `ter_wrong_only_batch4_v1/mistral_7b_instruct_v03` 为准。Gemma 已完成，三组511/1138（44.90%）、518/1138（45.52%）、620/1138（54.48%），以 `ter_wrong_only_batch4_memory_v3/gemma2_9b_it` 为准。二者分别由原审计器独立重审，记录为来源引用，不改写协议标签。
>
> Yi 原 v3 预检因 EOS 配置与聊天模板不符反复失败：模型 generation_config 为 ID 2（endoftext），tokenizer 聊天 EOS 为 ID 7（im_end）。独立服务包装器 `serve_benchmark_memory_control_yi_eos_v1.py` 在内存中使用 tokenizer EOS；原权重、tokenizer 文件、官方模板及原服务源码保持原文件。修复后4道合成题均在4 token结束，正式32-token基线已提交。v4的基线与当前运行环境完全一致；v5按字节接续，并逐题验证部分文件的原始前缀，保护所有已提交结果。
>
> 当前入口 `run_three_model_wrong_only_memory_v5.py`，协议 `ter-wrong-only-batch4-memory-v5`，结果根目录 `questionbank_eval/model_benchmarks/ter_wrong_only_batch4_memory_v5`，覆盖配置 `90-wrong-only-memory-v5.conf`。只继续 Yi，完成表通过 `completed_model_references.json` 引用上述 Mistral/Gemma。`preflight_wrong_only_memory_v5.py` 分别核验8001 Critic和8002 Yi的真实包装器哈希，直接保留接口原始元数据；`summarize_ter_wrong_only_memory_v5.py` 验证该契约、EOS7、迁移前缀和原完整错题审计。仍采用4题批量、回答32、证据Critic640、一次JSON修复、仅TER错题、正确与不可评分题原样保留。备份和诊断位于 `deployment_patches/20261020_gemma_shared_vram`。

以下为历史 v3 的显存修复记录。

> 2026-10-20：Gemma 在长输入预检后保留的闲置 CUDA 缓存使证据 Critic 的4题请求显存不足。复现同批4题，释放闲置缓存前 HTTP500，释放后 HTTP200。另发现原模拟压力包按 Qwen tokenizer 计1895 token，超过实际 Critic640-token上限；真实上限内635-token模拟包的候选4题重答通过。当前执行器为独立协议 `ter-wrong-only-batch4-memory-v3`，入口 `run_three_model_wrong_only_memory_v3.py`，新结果目录 `questionbank_eval/model_benchmarks/ter_wrong_only_batch4_memory_v3/`，配置 `70-wrong-only-memory-v3.conf`。两服务每次请求结束释放闲置 CUDA 缓存；仅在人工压力探测时用实际 Critic640-token预算内的包，正式生成器和真实 Critic 输出不变。
>
> Mistral 已完成的结果继续以本页原 v1 目录为准，由新目录 `completed_model_references.json` 记录来源、协议及哈希，不改写为 v3 预测。Gemma 四份完整基线经契约、ID、冻结证据和哈希验证后按字节复用，v3 错题阶段从空目录开始；v2 没有正式精修预测，已备份停止。Yi 由 v3 自动接力。新审计入口为 `summarize_ter_wrong_only_memory_v3.py`，额外验证两服务显存策略和 `stress_budget` 预检契约；原审计器、v1/v2及历史生成代码保持原文件不变。权重、量化、官方模板、题目、冻结证据、4题批量及32/640预算保持既定配置。故障复现、备份和部署验证位于 `deployment_patches/20261020_gemma_shared_vram/`。

2026-10-20 用户将任务改为昨天四模型的链路：候选 TER 直接回答 y0 → 离线选出 TER 错题 → 固定 Qwen3.5+LoRA v3 证据 Critic → 候选重答。TER 正确题和不可计分题原样保留。无全题回答 Critic 阶段。

继续采用已确认的 4 题批量、回答上限 32 token，证据 Critic 上限 640 token。昨天旧四模型的证据批量为 2；本版本保留今天用户确认的 4 题批量。两者的生成数值可能有差异，不把旧精修预测混入本版本。

模型顺序：Mistral-7B-Instruct-v0.3、Gemma-2-9B-it、Yi-1.5-9B-Chat。只汇总普通向量 RAG、TER、TER+EviCR v3 三组，分母均为 1,138（原始题 566+596，可评分 554+584）。保留历史格式失败默认 A 的显式标记；标准答案只用于离线评分与错题筛选。

正式协议 `ter-wrong-only-batch4-v1`，结果目录 `/home/ubuntu/tcm_rag/alignrag_tcm/questionbank_eval/model_benchmarks/ter_wrong_only_batch4_v1/`。入口 `run_three_model_wrong_only.py`；systemd 服务 `multimodel-exam-benchmark.service`，新覆盖配置 `50-ter-wrong-only.conf`。

Mistral 四份 32-token/4题 dense、TER 结果从严格 replay_v1 校验后按字节复用，记录 `baseline_reuse.json`。当前候选官方模板、检索证据、原始模型与 Critic 适配器保持原配置。128-token adapted_v2、两轮回答 Critic 的中间预测和探测输出均留在各自目录。

每个候选启动后检查官方模板与长输入；Critic 同驻显存后仅探测 TER 错题的 640-token 证据 Critic 和 32-token 候选重答，以及4题最大证据包。没有回答 Critic 探测或评分阶段。每阶段冻结生成契约、逐题原子提交；重试先恢复当前模型，验证完成后才接力下一个。

完成时用 `summarize_ter_wrong_only.py` 独立核对两年三组分数、全部 ID、TER 错题集合、y0、正确题原样保留、32/640预算、固定4题契约及代码/证据哈希。审计通过后才按已有规则清理可重新下载的候选权重，保护原 Qwen3.5、Critic 适配器、冻结证据和历史输出。

原两轮任务已停止，结果原位保留，并备份到 `deployment_patches/20261020_ter_wrong_only_batch4_v1/backup/`。远端自动接力、失败重启和断点恢复独立于本地电脑运行。
