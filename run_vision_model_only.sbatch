#!/bin/bash
#SBATCH --job-name=vision_model_only18
#SBATCH --partition=debug
#SBATCH --account=ACCOUNT
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=$FORESIGHT_ROOT/evaluation/output_vision_model_only/slurm_%j.log
set -euo pipefail
REPO=$FORESIGHT_ROOT
OUT=${OUT:-$REPO/evaluation/output_vision_model_only}
PY=${PY:-python}
PLOT_PY=${PLOT_PY:-$HOME/miniforge3/bin/python3}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export FORESIGHT_DIR=$REPO/foresight
if [[ -z ${SLURM_JOB_ID:-} && ${ALLOW_LOGIN_DIAGNOSTIC:-0} != 1 ]]; then
    echo 'Outside SLURM: explicit ALLOW_LOGIN_DIAGNOSTIC=1 required.' >&2
    exit 1
fi
mkdir -p "$OUT"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader
pids=()
extra=()
if [[ ${REALTIME_DIAGNOSTIC:-1} == 1 ]]; then
    extra+=(--realtime)
else
    extra+=(--lockstep)
fi
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu "$PY" "$REPO/evaluation/vision_model_only.py" run \
        --out "$OUT" --shard "$gpu" --arm "${TRIGGER_ARM:-raw}" \
        --wall-budget-s "${RUN_BUDGET_S:-0}" "${extra[@]}" >> "$OUT/g${gpu}.log" 2>&1 &
    pids+=("$!")
done
trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM EXIT
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
trap - INT TERM EXIT
if [[ $rc != 0 ]]; then
    echo 'At least one worker failed; inspect shard logs. No complete plot produced.' >&2
    exit "$rc"
fi
if [[ ${REALTIME_DIAGNOSTIC:-1} != 1 ]]; then
    "$PLOT_PY" "$REPO/evaluation/vision_model_only.py" plot --out "$OUT"
fi
echo "Finished $(date -Is)"