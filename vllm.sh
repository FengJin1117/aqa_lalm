# 使用conda环境：vllm

CUDA_VISIBLE_DEVICES=7 VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 32768