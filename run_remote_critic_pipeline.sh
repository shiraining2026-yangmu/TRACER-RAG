#!/usr/bin/env bash
# Build R4-retrieved Critic supervision and train a Qwen LoRA adapter.
# Run only after copying alignrag_tcm/ and critic_seed_3000.jsonl to the remote host.
set -euo pipefail

# Required remote paths. Defaults reflect the documented R4 layout and can be
# overridden through environment variables without editing this file.
: "${TCM_ROOT:=/home/ubuntu/tcm_rag}"
: "${AEK_ROOT:=${TCM_ROOT}/aek_baseline_r1}"
: "${AEK_INPUTS_DIR:=${AEK_ROOT}/inputs_r2}"
: "${R4_RUNNER:=${AEK_ROOT}/src/run_aek_hybrid_r4.py}"
: "${CRITIC_ROOT:=${TCM_ROOT}/alignrag_tcm}"
: "${WORK_DIR:=${AEK_ROOT}/runs/alignrag_critic_3k_v1}"
: "${BGE_M3_PATH:=/home/ubuntu/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/main}"
: "${PYTHON_BIN:=${TCM_ROOT}/.venvs/tier2_qwen_transformers457/bin/python}"
: "${WEAK_API_BASE:=http://127.0.0.1:8000}"
: "${WEAK_MODEL:=/home/ubuntu/tcm_rag/models/Qwen3.5-9B}"
: "${TEACHER_API_BASE:=${WEAK_API_BASE}}"
: "${TEACHER_MODEL:=${WEAK_MODEL}}"
: "${TEACHER_API_KEY_ENV:=}"
: "${CRITIC_BASE_MODEL:=${WEAK_MODEL}}"
: "${R4_RETRIEVAL_SOURCE:=}"

SEEDS="${CRITIC_ROOT}/data/critic_seed_3000.jsonl"
RETRIEVAL_DIR="${WORK_DIR}/r4_retrieval"
INDEX_DIR="${WORK_DIR}/r4_index"
WEAK_OUT="${WORK_DIR}/01_weak.jsonl"
TEACHER_OUT="${WORK_DIR}/02_teacher.jsonl"
CRITIQUE_OUT="${WORK_DIR}/03_critique.jsonl"
SFT_DIR="${WORK_DIR}/critic_sft"
ADAPTER_DIR="${WORK_DIR}/qwen_critic_lora"

for path in "${SEEDS}" "${R4_RUNNER}" "${CRITIC_ROOT}/materialize_critic_retrieval.py" "${CRITIC_ROOT}/generate_critic_supervision.py" "${CRITIC_ROOT}/export_critic_sft.py" "${CRITIC_ROOT}/train_critic_lora.py"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing executable Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ ! -e "${WORK_DIR}" ]] || { echo "WORK_DIR already exists: ${WORK_DIR}" >&2; exit 2; }
mkdir -p "${WORK_DIR}"

# 1. R4 retrieval. A hash-verified existing R4 run can be reused to avoid
# rebuilding the identical Build index. In either mode, source answers are never
# copied to the Critic input.
if [[ -n "${R4_RETRIEVAL_SOURCE}" ]]; then
  "${PYTHON_BIN}" "${CRITIC_ROOT}/materialize_critic_retrieval.py" \
    --seeds "${SEEDS}" --r4-retrieval "${R4_RETRIEVAL_SOURCE}" \
    --output "${RETRIEVAL_DIR}/generation_results.jsonl" --top-k 3
else
  "${PYTHON_BIN}" "${R4_RUNNER}" \
    --queries "${SEEDS}" \
    --build-edges "${AEK_INPUTS_DIR}/build_edges.aek.v1.jsonl" \
    --applications "${AEK_INPUTS_DIR}/applications.aek.v1.jsonl" \
    --knowledge "${AEK_INPUTS_DIR}/knowledge.aek.v1.jsonl" \
    --run-dir "${RETRIEVAL_DIR}" \
    --index-dir "${INDEX_DIR}" \
    --embedding-model "${BGE_M3_PATH}" \
    --split build \
    --phase retrieve \
    --top-k 3 \
    --bm25-candidates 100 \
    --dense-candidates 100
fi

# 2. Test the three generation phases with local Qwen first. For final teacher
# data, set TEACHER_API_BASE=https://api.deepseek.com, TEACHER_MODEL=<model>,
# export the key named by TEACHER_API_KEY_ENV, then run this pipeline anew.
"${PYTHON_BIN}" "${CRITIC_ROOT}/generate_critic_supervision.py" \
  --phase weak --input "${RETRIEVAL_DIR}/generation_results.jsonl" --output "${WEAK_OUT}" \
  --weak-api-base "${WEAK_API_BASE}" --weak-model "${WEAK_MODEL}"

"${PYTHON_BIN}" "${CRITIC_ROOT}/generate_critic_supervision.py" \
  --phase teacher --input "${WEAK_OUT}" --output "${TEACHER_OUT}" \
  --weak-api-base "${WEAK_API_BASE}" --weak-model "${WEAK_MODEL}" \
  --teacher-api-base "${TEACHER_API_BASE}" --teacher-model "${TEACHER_MODEL}" \
  --teacher-api-key-env "${TEACHER_API_KEY_ENV}"

"${PYTHON_BIN}" "${CRITIC_ROOT}/generate_critic_supervision.py" \
  --phase critique --input "${TEACHER_OUT}" --output "${CRITIQUE_OUT}" \
  --weak-api-base "${WEAK_API_BASE}" --weak-model "${WEAK_MODEL}" \
  --teacher-api-base "${TEACHER_API_BASE}" --teacher-model "${TEACHER_MODEL}" \
  --teacher-api-key-env "${TEACHER_API_KEY_ENV}"

# 3. Strictly parse teacher JSON critiques. The exported student prompt has
# q + D(q) + y_weak only. y_strong is excluded at this boundary.
"${PYTHON_BIN}" "${CRITIC_ROOT}/export_critic_sft.py" \
  --input "${CRITIQUE_OUT}" --output-dir "${SFT_DIR}" --min-accepted 2500

# 4. Train the small Qwen Critic as a QLoRA adapter.
"${PYTHON_BIN}" "${CRITIC_ROOT}/train_critic_lora.py" \
  --model-name-or-path "${CRITIC_BASE_MODEL}" \
  --train-file "${SFT_DIR}/critic_sft_train.jsonl" \
  --validation-file "${SFT_DIR}/critic_sft_validation.jsonl" \
  --output-dir "${ADAPTER_DIR}" \
  --quantization 4bit --max-length 4096 --epochs 2 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 --lora-r 32 --lora-alpha 64 --gradient-checkpointing
