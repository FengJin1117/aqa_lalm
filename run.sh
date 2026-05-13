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
  --max_samples 1000  > logs/cap:bagpiper_ans:deepseek_caption_only.log 2>&1 &

python compare_results.py \
    -a outputs/QwenOmni/direct_aqa_results.json \
    -b outputs/QwenOmni/caption_only_results.json \
    --name_a "Direct AQA" --name_b "Based on Qwen3 Captioner Caption"