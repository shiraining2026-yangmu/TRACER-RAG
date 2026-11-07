#!/usr/bin/env bash
# Sequential candidate models; frozen inputs; fixed Qwen3.5+adapter Critic.
set -Eeuo pipefail
ROOT="${TCM_ROOT:-/home/ubuntu/tcm_rag}"
APP="${ROOT}/alignrag_tcm"
PY="${PYTHON_BIN:-${ROOT}/.venvs/tier2_qwen_transformers457/bin/python}"
BENCH="${BENCHMARK_OUTPUT_ROOT:-${APP}/questionbank_eval/model_benchmarks}"
LOGS="${BENCH}/logs"
MODEL_ROOT="${ROOT}/models_benchmark"
GEN_BASE=http://127.0.0.1:8002
CRITIC_BASE=http://127.0.0.1:8001
REFDIR="${APP}/questionbank_eval/model_benchmarks/_reference"
ADAPTED=${BENCHMARK_ADAPTED_V2:-0}
CRITIC_MODEL=qwen-critic-repair-v2
CRITIC_BASE_MODEL="${ROOT}/models/Qwen3.5-9B"
CRITIC_ADAPTER="${APP}/runs/alignrag_critic_3k_v1/selective_repair_v2/qwen_critic_lora_balanced_v2_len2560"
MODELS=(
 'qwen2_5_7b_instruct|Qwen/Qwen2.5-7B-Instruct|0|'
 'qwen3_8b|Qwen/Qwen3-8B|0|Qwen/Qwen3-8B'
 'internlm3_8b_instruct|internlm/internlm3-8b-instruct|1|Shanghai_AI_Laboratory/internlm3-8b-instruct'
 'meta_llama_3_1_8b_instruct|meta-llama/Llama-3.1-8B-Instruct|0|LLM-Research/Meta-Llama-3.1-8B-Instruct'
 'mistral_7b_instruct_v03|mistralai/Mistral-7B-Instruct-v0.3|0||dense_ter_evicr'
 'gemma2_9b_it|google/gemma-2-9b-it|0|google/gemma-2-9b-it|dense_ter_evicr'
 'yi15_9b_chat|01-ai/Yi-1.5-9B-Chat|1|01ai/Yi-1.5-9B-Chat|dense_ter_evicr'
)
if [[ "$ADAPTED" == 1 ]]; then
 MODELS=(
  'mistral_7b_instruct_v03|mistralai/Mistral-7B-Instruct-v0.3|0||dense_ter_evicr'
  'gemma2_9b_it|google/gemma-2-9b-it|0|google/gemma-2-9b-it|dense_ter_evicr'
  'yi15_9b_chat|01-ai/Yi-1.5-9B-Chat|0|01ai/Yi-1.5-9B-Chat|dense_ter_evicr'
 )
fi
mkdir -p "${LOGS}" "${MODEL_ROOT}"

json_state() {
 local directory=$1 status=$2 phase=$3
 mkdir -p "$directory"
 printf '{"status":"%s","phase":"%s","updated_at":"%s","pid":%s}\n' "$status" "$phase" "$(date -Is)" "$$" > "${directory}/state.json.tmp"
 mv "${directory}/state.json.tmp" "${directory}/state.json"
}

stop_service() {
 local pidfile=$1 expected=$2
 "${PY}" - "$pidfile" "$expected" <<'PY'
import os,signal,sys,time
from pathlib import Path
path=Path(sys.argv[1])
if not path.exists(): sys.exit(0)
pid=int(path.read_text().strip())
try:
    if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':
        path.unlink(missing_ok=True)
        sys.exit(0)
    command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if sys.argv[2] not in command:
        raise ValueError('Stale service PID belongs to another process; refusing to stop it')
    os.kill(pid,signal.SIGTERM)
    for _ in range(30):
        if not Path(f'/proc/{pid}').exists(): break
        if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z': break
        time.sleep(.5)
    else: os.kill(pid,signal.SIGKILL)
except ProcessLookupError: pass
except FileNotFoundError: pass
path.unlink(missing_ok=True)
PY
}

stop_owned() {
 stop_service "${LOGS}/generator.pid" serve_single_generator.py
 stop_service "${LOGS}/critic.pid" serve_qwen_critic_adapter.py
}

wait_models() {
 local base=$1 expected=$2 pid=$3
 for _ in $(seq 1 120); do
  if curl -fsS "${base}/v1/models" 2>/dev/null | grep -Fq "$expected"; then return 0; fi
  kill -0 "$pid" 2>/dev/null || return 1
  sleep 5
 done
 return 1
}

start_generator() {
 stop_service "${LOGS}/generator.pid" serve_single_generator.py
 local args=("${PY}" "${APP}/serve_single_generator.py" --model-path "${model_path}" --model-id "${model_id}" --host 127.0.0.1 --port 8002)
 if [[ "$trust" == 1 ]]; then args+=(--trust-remote-code); fi
 if [[ "$slug" == internlm3_8b_instruct ]]; then
  # Official InternLM custom code targets Transformers 4.x; keep a separate
  # small overlay without altering the frozen Qwen3.5 runtime.
  local compat="${ROOT}/.venvs/internlm3_compat"
  if [[ ! -f "${compat}/ready" ]]; then
   mkdir -p "$compat"
   "${PY}" -m pip install --no-cache-dir --target "$compat" --no-deps 'transformers==4.48.3' 'tokenizers==0.21.4' 'huggingface-hub==0.36.0' 'peft==0.14.0'
   touch "${compat}/ready"
  fi
  PYTHONPATH="$compat" "${args[@]}" >> "${work}/logs/generator_service.log" 2>&1 < /dev/null &
 else
  "${args[@]}" >> "${work}/logs/generator_service.log" 2>&1 < /dev/null &
 fi
 local pid=$!
 echo "$pid" > "${LOGS}/generator.pid"
 wait_models "$GEN_BASE" "$model_id" "$pid"
}

start_critic() {
 stop_service "${LOGS}/critic.pid" serve_qwen_critic_adapter.py
 "${PY}" "${APP}/serve_qwen_critic_adapter.py" --model "$CRITIC_BASE_MODEL" --adapter "$CRITIC_ADAPTER" --generator-model-id qwen-generator-unused --critic-model-id "$CRITIC_MODEL" --host 127.0.0.1 --port 8001 >> "${LOGS}/critic_service.log" 2>&1 < /dev/null &
 local pid=$!
 echo "$pid" > "${LOGS}/critic.pid"
 wait_models "$CRITIC_BASE" "$CRITIC_MODEL" "$pid"
}

grade() {
 "${PY}" "${APP}/grade_single_results.py" --results "$1" --reference-grading "$2" --output-dir "$3" --label "$4"
}

run_model() {
 IFS='|' read -r slug repo trust mirror protocol <<< "$1"
 protocol=${protocol:-full}
 case "$protocol" in
  full) local summary_arms=(no_rag dense ter evicr);;
  dense_ter_evicr) local summary_arms=(dense ter evicr);;
  *) echo "Unknown experiment protocol: $protocol" >&2; return 1;;
 esac
 work="${BENCH}/${slug}"
 model_path="${MODEL_ROOT}/${slug}"
 model_id="benchmark-${slug}"
 local generation_args=() refinement_args=() baseline_batch=4 refinement_batch=2 max_tokens=32
 if [[ "$ADAPTED" == 1 ]]; then
  max_tokens=128
  generation_args=(--max-new-tokens 128 --invalid-choice-policy mark_invalid)
  refinement_args=(--generator-max-tokens 128 --invalid-choice-policy mark_invalid)
 fi
 mkdir -p "${work}"/{logs,no_rag,dense_rag,ter_direct,evicr_v3,grading}
 trap 'code=$?; json_state "$work" retry_pending "${phase:-unknown}"; stop_owned; exit "$code"' ERR
 trap 'stop_owned; exit 143' TERM INT
 if [[ -f "${work}/summary.json" ]]; then
  "${PY}" "${APP}/summarize_multimodel_benchmark.py" --work-dir "$work" --slug "$slug" --arms "${summary_arms[@]}"
 else
  phase=download_model
  json_state "$work" running "$phase"
  local download=("${PY}" "${APP}/download_benchmark_model.py" --repo "$repo" --path "$model_path" --manifest "${work}/model_source.json")
  [[ -z "$mirror" ]] || download+=(--modelscope "$mirror")
  "${download[@]}"
  "${PY}" - "$work" "$repo" "$CRITIC_BASE_MODEL" "$CRITIC_ADAPTER" "$protocol" "$max_tokens" "$ADAPTED" <<'PY'
import json,sys
from pathlib import Path
value={'repo':sys.argv[2], 'critic_base':sys.argv[3], 'critic_adapter':sys.argv[4],
       'protocol':sys.argv[5],
       'chat_template_compatibility':'official template; merge leading system into first user only if template rejects system',
       'answer_key_used_for_model_input':False, 'candidate_quantization':'4-bit NF4 double quantization',
       'evicr_protocol':'TER wrong-only v3 repair; preserves TER correct answers using offline labels',
       'generation':{'do_sample':False,'max_new_tokens':int(sys.argv[6]),'enable_thinking':False},
       'deployment_protocol':'adapted_v2' if sys.argv[7]=='1' else 'legacy',
       'invalid_output_scoring':'wrong; never default A' if sys.argv[7]=='1' else 'legacy default A',
       'critic_max_new_tokens':640,'baseline_batch_size':4,'evicr_batch_size':2}
Path(sys.argv[1],'manifest.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
PY
  phase=generator_baselines
  json_state "$work" running "$phase"
  start_generator
  if [[ "$ADAPTED" == 1 ]]; then
   phase=deployment_preflight
   json_state "$work" running "$phase"
   "${PY}" "${APP}/preflight_generator_deployment.py" --api-base "$GEN_BASE" --model "$model_id" --output "${work}/deployment_preflight.json"
   baseline_batch=$("${PY}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["baseline_batch_size"])' "${work}/deployment_preflight.json")
   refinement_batch=$("${PY}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["refinement_batch_size"])' "${work}/deployment_preflight.json")
   generation_args+=(--runtime-contract "${work}/deployment_preflight.json")
  fi
  if [[ "$protocol" == full ]]; then
   "${PY}" "${APP}/run_questionbank_no_rag.py" --questions "${APP}/questionbank_eval/questions_2022_mandatory_choice_v3_answer_blind.jsonl" "${APP}/questionbank_eval/questions_2021_mandatory_choice_v3_answer_blind.jsonl" --output-dir "${work}/no_rag" --api-base "$GEN_BASE" --generator-model "$model_id" --batch-size 4 --resume
   for year in 2022 2021; do
    grade "${work}/no_rag/${year}_mandatory_choice_v3_answer_blind/no_rag_results.jsonl" "${REFDIR}/reference_${year}.csv" "${work}/grading/no_rag_${year}" "no_rag_${year}"
   done
  fi
  for arm in dense ter; do
   phase="${arm}_generation"
   json_state "$work" running "$phase"
   for year in 2022 2021; do
    local input dir
    if [[ "$arm" == dense ]]; then
     input="${APP}/questionbank_eval/dense_chunk_rag_v1/${year}_mandatory_choice_v3_answer_blind/retrieval.jsonl"; dir=dense_rag
    else
     input="${APP}/questionbank_eval/mandatory_choice_v3_${year}/alignrag_results.jsonl"; dir=ter_direct
    fi
    "${PY}" "${APP}/run_frozen_evidence_generation.py" --arm "$arm" --evidence-input "$input" --output "${work}/${dir}/${year}/${dir}_results.jsonl" --api-base "$GEN_BASE" --generator-model "$model_id" --batch-size "$baseline_batch" "${generation_args[@]}" --resume
    grade "${work}/${dir}/${year}/${dir}_results.jsonl" "${REFDIR}/reference_${year}.csv" "${work}/grading/${arm}_${year}" "${arm}_${year}"
   done
  done
  phase=evicr_v3
  json_state "$work" running "$phase"
  start_critic
  if [[ "$ADAPTED" == 1 ]]; then
   "${PY}" "${APP}/preflight_generator_deployment.py" --api-base "$GEN_BASE" --model "$model_id" --critic-present --output "${work}/refinement_deployment_preflight.json"
   refinement_batch=$("${PY}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["refinement_batch_size"])' "${work}/refinement_deployment_preflight.json")
   refinement_args+=(--runtime-contract "${work}/refinement_deployment_preflight.json")
  fi
  for year in 2022 2021; do
   local src="${work}/ter_direct/${year}/ter_direct_results.jsonl" scored="${work}/grading/ter_${year}/grading.csv" out="${work}/evicr_v3/${year}"
   "${PY}" "${APP}/run_evidence_refinement_wrong_questions.py" --source-results "$src" --grading-csv "$scored" --output-dir "$out" --wrong-from alignrag --generator-api-base "$GEN_BASE" --critic-api-base "$CRITIC_BASE" --generator-model "$model_id" --critic-model "$CRITIC_MODEL" --batch-size "$refinement_batch" "${refinement_args[@]}" --resume
   "${PY}" "${APP}/replace_targeted_inference_rows.py" --original "$src" --replacement "${out}/evidence_refinement_results.jsonl" --grading-csv "$scored" --output "${out}/evicr_v3_merged_results.jsonl"
   grade "${out}/evicr_v3_merged_results.jsonl" "${REFDIR}/reference_${year}.csv" "${work}/grading/evicr_${year}" "evicr_v3_${year}"
  done
  stop_owned
  phase=validate_summary
  json_state "$work" running "$phase"
  "${PY}" "${APP}/summarize_multimodel_benchmark.py" --work-dir "$work" --slug "$slug" --arms "${summary_arms[@]}"
 fi
 "${PY}" - "$model_path" "$MODEL_ROOT" <<'PY'
import shutil,sys
from pathlib import Path
target=Path(sys.argv[1]); root=Path(sys.argv[2]).resolve()
if target.is_symlink() or target.resolve().parent!=root or target.name not in {'qwen2_5_7b_instruct','qwen3_8b','internlm3_8b_instruct','meta_llama_3_1_8b_instruct','mistral_7b_instruct_v03','gemma2_9b_it','yi15_9b_chat'}:
    raise ValueError('Unsafe candidate cleanup target')
if target.exists(): shutil.rmtree(target)
PY
 touch "${work}/model_deleted_after_verified_summary"
 json_state "$work" complete complete
}

if [[ "${1:-}" == --model ]]; then
 run_model "$2"
 exit 0
fi

exec 9> "${BENCH}/supervisor.lock"
flock -n 9 || exit 0
echo "$$" > "${BENCH}/supervisor.pid"
"${PY}" "${APP}/build_multimodel_reference_grading.py" --source "${REFDIR}/source_2021.csv" "${REFDIR}/source_2022.csv" --output-dir "$REFDIR"
for record in "${MODELS[@]}"; do
 slug=${record%%|*}
 success=0
 for attempt in 1 2 3; do
  echo "[$(date -Is)] ${slug}: attempt ${attempt}"
  if bash "$0" --model "$record"; then success=1; break; fi
  echo "[$(date -Is)] ${slug}: retry pending; completed answers retained"
  sleep 30
 done
 if [[ "$success" == 0 ]]; then
  json_state "${BENCH}/${slug}" failed retry_exhausted
  # Keep only this incomplete candidate on disk. The systemd retry resumes
  # it before any other model can be downloaded into the limited system disk.
  exit 1
 fi
done
echo "[$(date -Is)] all queued model benchmarks complete"
