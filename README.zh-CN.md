# TRACER-RAG

面向中医文本问答的**可追溯关系证据检索 TER**与**证据评议驱动的回答精修 EviCR**研究代码。

[English](README.md) · [复现步骤](docs/REPRODUCIBILITY.md) · [数据格式](docs/DATA_CONTRACTS.md) · [代码索引](docs/CODE_STRUCTURE.md)

## 当前协议

TER 将历史应用 A、可复用知识 K 及支撑关系 E 组织为检索单元，经过词法/稠密混合召回与软关系重排，提供固定证据 D(q)。

```text
离线：q + D(q) -> 教师证据标签 -> q、D(q) 输入的 Critic 微调
在线：q + D(q) -> Critic -> 结构化证据意见
      q + D(q) -> 回答模型 -> 初答 y0
      q + D(q) + y0 + 证据意见 -> 回答模型 -> 单轮修订 y1
```

**离线和在线 Critic 均只读取 q、D(q)，不读取当前初答、弱回答或答案键。** D(q) 中的历史应用回答属于检索材料，与当前问题的 y0 不同。教师标注、SFT 导出与公开在线入口复用同一组 system/user 提示构造函数。

输出字段为 `evidence_status`、`evidence_actions`、`knowledge_supplements`、`change_support`、`answer_guidance`；模型知识补充与检索证据明确区分。

## 无需密钥的演示

Python 3.10+：

```bash
python examples/offline_demo.py --output-dir outputs/demo
pip install -r requirements-test.txt
python -m pytest -q
python scripts/check_release.py
```

演示使用非医学的符号样例，贯通 Build 监督准备、SFT 导出、训练 dry-run 与推理提示预览，不调用模型或付费接口。人工构造的演示标签不代表实验结果。

完整流程见[复现文档](docs/REPRODUCIBILITY.md)。正式模型运行需自行准备 CUDA/PyTorch、与模型架构兼容的 Transformers/PEFT 版本、模型文件及已获许可的数据。`requirements.txt` 不是完整锁定的实验环境。

## 本次发布范围

- TER 构库、混合检索、可配置重排与 Dev 权重搜索。
- q、D(q) 协议的监督构建、标签校验、SFT 导出、QLoRA 训练与服务。
- 任意规模的完整 Test 推理入口 `run_evicr.py`；禁用知识补充时复用同一份 Critic 输出。
- 证据忠实性评估脚本、配对统计、合成样例、自动测试与发布检查。
- 已核查的 Dev 权重搜索汇总，以及源文件/修改记录和校验和。

保留的回答感知 Critic 和错题定向修复脚本已在[代码索引](docs/CODE_STRUCTURE.md)标为历史流程，不能当作当前 EviCR 的正式入口。历史上用答案键挑选错题的结果属于条件性修复诊断，不能直接解释为盲测全量增益。

仓库未包含原始病例、原始题库、答案键、逐题输出、SFT 数据、模型/Adapter、密钥、论文未公开稿或审稿材料。因而这是代码与复现流程发布，并非所有论文数值的一键完整复现包。脚本可运行、单元测试通过与实验有效性是不同结论；尚未核验的实验不标为已完成。参见[数据说明](DATA_AVAILABILITY.md)、[发布范围](docs/RELEASE_SCOPE.md)和[权利说明](RIGHTS.md)。

尚未指定软件复用许可证；模型及数据各自的使用条款仍须遵守。该代码用于研究，不是经过临床验证的诊疗工具。
