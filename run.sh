# 给出python的nuhup命令，运行test_qwen2_audio.py，并将输出保存到test_qwen2_audio.log文件中，同时在后台运行。
# nohup python test_qwen2_audio.py > logs/test_qwen2_audio.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python run_eval.py \
#     --model "Qwen2Audio" \
#     --max_samples 1000 > logs/eval.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python run_eval.py \
#     --model "QwenOmni" \
#     --prompt_type "caption_aqa" \
#     --max_samples 1000 > logs/qwenomni_caption.log 2>&1 &

# # 命令：查看当前用户占用GPU的情况
# nvidia-smi -u $USER

python compare_results.py \
    -a outputs/20260412_034514_results.json \
    -b outputs/QwenOmni/20260413_105028_results.json \
    --name_a "Direct AQA" --name_b "Caption-Reasoning-Answer"

# 
CUDA_VISIBLE_DEVICES=6 nohup python run_eval.py \
  --model QwenOmni \
  --prompt_type caption_only \
  --max_samples 1000  > logs/cap:bagpiper_ans:qwenomni_caption_only.log 2>&1 &


CUDA_VISIBLE_DEVICES=6 nohup python run_eval.py \
  --model deepseek \
  --prompt_type caption_only \
  --max_samples 1000  > logs/cap:qwne3A3B_ans:deepseek_caption_only.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python run_eval.py \
  --model QwenOmni \
  --prompt_type caption_only \
  --caption_path /data2/fwh/audio_caption/outputs/captions.jsonl \
  --max_samples 1000  > logs/cap:qwne25omni_ans:qwenomni_caption_only.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python run_eval.py \
  --model QwenOmni \
  --prompt_type caption_only \
  --caption_path /data2/fwh/audio_caption/outputs/captions.jsonl \
  --max_samples 1000  > logs/cap:qwne25omni_ans:qwenomni_caption_only.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python run_eval.py \
  --model QwenOmni \
  --prompt_type caption_only \
  --caption_path data/mmau-test-mini-shuffled-with-choice.json \
  --max_samples 1000  > logs/cap:empty_ans:qwenomni_caption_only_shuffle.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python run_eval.py \
  --benchmark_mode caption_qa \
  --caption_path /data2/fwh/audio_caption/outputs/captions_empty.jsonl \
  --max_samples 1000  > logs/cap:empty_ans:qwenomni_caption_only_shuffle.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python run_eval.py \
  --benchmark_mode caption_qa \
  --text_model QwenOmni \
  --caption_path /data2/fwh/audio_caption/outputs/captions_empty.jsonl \
  --max_samples 1000  > logs/cap:empty_ans:qwenomni_caption_only_shuffle_fixbug.log 2>&1 &

python compare_results.py \
    -a outputs/QwenOmni/direct_aqa_results.json \
    -b outputs/QwenOmni/caption_only_results.json \
    --name_a "Direct AQA" --name_b "Based on Qwen3 Captioner Caption"

# MMSU

CUDA_VISIBLE_DEVICES=6 python run_eval.py \
  --mode caption_qa \
  --benchmark MMSU
  --text_model QwenOmni \
  --caption_path /data2/fwh/audio_caption/outputs/captions_empty.jsonl \
  --max_samples 1000  > logs/cap:empty_ans:qwenomni_caption_only_shuffle_fixbug.log 2>&1 &

# Qwen 35 omni caption

CUDA_VISIBLE_DEVICES=4 python run_eval.py \
  --benchmark_mode caption_qa \
  --text_model QwenOmni \
  --caption_path /data2/fwh/audio_caption/outputs/mmau_captions_qwen35omniplus.jsonl \
  --max_samples 1000  > logs/cap:qwen35omni_ans:qwenomni_caption_only.log 2>&1 &


# 4B
CUDA_VISIBLE_DEVICES=4 python run_eval.py \
  --benchmark_mode caption_qa \
  --text_model Qwen3-4B-Instruct-2507 \
  --caption_path /data2/fwh/audio_caption/outputs/captions_empty.jsonl \
  --max_samples 1000  > logs/cap:empty_ans:wen3-4B-instruct_caption_only.log 2>&1 &


CUDA_VISIBLE_DEVICES=4 python run_eval.py \
  --benchmark_mode caption_qa \
  --caption_path /data2/fwh/audio_caption/outputs/captions_empty.jsonl \
  --max_samples 1000 > logs/cap:empty_ans:wen3-4B-instruct_caption_only.log 2>&1 &

# Qwen3-4B-Instruct-2507

## 
CUDA_VISIBLE_DEVICES=4 python run_eval.py \
  --benchmark_mode caption_qa \
  --vllm_gpu 6 \
  --caption_path data/captions/mmau_captions_qwen25omni.jsonl \
  --max_samples 1000  > logs/cap:qwen25omni_ans:qwen3-4B-instruct_caption_only.log 2>&1 &

## captioner
python run_eval.py \
  --benchmark_mode caption_qa \
  --vllm_gpu 6 \
  --caption_path /data2/fwh/aqa_lalm/data/captions/mmau-test-mini-captions-captioner.jsonl \
  --max_samples 1000  > logs/cap:captioner_ans:qwen3-4B-instruct_caption_only.log 2>&1 &

python run_eval.py \
  --benchmark_mode caption_qa \
  --vllm_gpu 7 \
  --caption_path /data2/fwh/aqa_lalm/data/captions/mmau_captions_qwen35omniplus.jsonl \
  --max_samples 1000  > logs/cap:qwen35omniplus_ans:qwen3-4B-instruct_caption_only.log 2>&1 &

## bagpiper


python run_eval.py \
  --benchmark_mode caption_qa \
  --vllm_gpu 7 \
  --caption_path /data2/fwh/aqa_lalm/data/captions/mmau_captions_bagpiper.jsonl \
  --max_samples 1000  > logs/cap:qwen35omniplus_ans:qwen3-4B-instruct_caption_only.log 2>&1 &