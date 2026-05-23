#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 vLLM deployment and benchmark entrypoint.
# Conda environment: vllm

set -euo pipefail

MODEL="Qwen/Qwen3-4B-Instruct-2507"
SERVED_MODEL="Qwen3-4B-Instruct-2507"
HOST="127.0.0.1"
BASE_PORT=8000
# GPUS=(4 5 6 7)
# 给多GPU用的（只管数目）
GPUS=(6 7)
EXP_DIR="exp/vllm_parallel"

serve_one() {
  local gpu="${1:-6}"
  local port="${2:-8000}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  VLLM_USE_MODELSCOPE=true \
  conda run --no-capture-output -n vllm vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${port}" \
    --max-model-len 32768 \
    --served-model-name "${SERVED_MODEL}" \
    --disable-log-requests
}

serve_many() {
  # 使用GPU的数据
  local count="${1:-2}"
  mkdir -p "${EXP_DIR}/logs"

  for idx in $(seq 0 $((count - 1))); do
    local gpu="${GPUS[$idx]}"
    local port=$((BASE_PORT + idx))
    echo "Starting ${SERVED_MODEL} on GPU ${gpu}, port ${port}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    VLLM_USE_MODELSCOPE=true \
    nohup conda run --no-capture-output -n vllm vllm serve "${MODEL}" \
      --host "${HOST}" \
      --port "${port}" \
      --max-model-len 32768 \
      --served-model-name "${SERVED_MODEL}" \
      --disable-log-requests \
      > "${EXP_DIR}/logs/vllm_gpu${gpu}_port${port}.log" 2>&1 &
  done
}

bench_serial() {
  conda run -n vllm python "${EXP_DIR}/bench_chat.py" \
    --base-urls "http://${HOST}:8000/v1" \
    --concurrency 1 \
    --num-requests 20 \
    --name serial_single_gpu
}

bench_concurrent() {
  for concurrency in 1 2 4; do
    conda run -n vllm python "${EXP_DIR}/bench_chat.py" \
      --base-urls "http://${HOST}:8000/v1" \
      --concurrency "${concurrency}" \
      --num-requests 40 \
      --name "concurrent_c${concurrency}_single_gpu"
  done
}

bench_scale() {
  for instances in 1 2 3 4; do
    local urls=()
    for idx in $(seq 0 $((instances - 1))); do
      urls+=("http://${HOST}:$((BASE_PORT + idx))/v1")
    done

    conda run -n vllm python "${EXP_DIR}/bench_chat.py" \
      --base-urls "${urls[@]}" \
      --concurrency 4 \
      --num-requests 80 \
      --name "scale_${instances}gpu"
  done
}

make_report() {
  conda run -n vllm python "${EXP_DIR}/make_report.py"
}

case "${1:-help}" in
  serve-one)
    serve_one "${2:-4}" "${3:-8000}"
    ;;
  serve-many)
    serve_many "${2:-4}"
    ;;
  bench-serial)
    bench_serial
    ;;
  bench-concurrent)
    bench_concurrent
    ;;
  bench-scale)
    bench_scale
    ;;
  report)
    make_report
    ;;
  help|*)
    cat <<'USAGE'
Qwen3-4B-Instruct-2507 vLLM deployment performance test

Usage:
  bash vllm.sh serve-one [gpu] [port]
  bash vllm.sh serve-many [instance_count]
  bash vllm.sh bench-serial
  bash vllm.sh bench-concurrent
  bash vllm.sh bench-scale
  bash vllm.sh report

Examples:
  bash vllm.sh serve-one 4 8000
  bash vllm.sh serve-many 4
  bash vllm.sh bench-serial
  bash vllm.sh bench-concurrent
  bash vllm.sh bench-scale
  bash vllm.sh report
USAGE
    ;;
esac
