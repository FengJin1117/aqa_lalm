# LALM AQA
这是一个统一 **LALM AQA推理封装**项目。支持各类Audio QA Benchmark的评测。
输入格式：audio + question

## 📦支持的模型
- ✅Qwen2.5-Omni-7B
- ✅Qwen2-Audio-7B-Instruct

外部模型API：
- ✅️DeepSeek API（仅支持纯文本回答）
这里需要：`pip install -e /data2/fwh/llm_processor`。这个包还在开发当中。

## 支持的Benchmark
- ✅️MMAU

## 支持的问答方式
- ✅️AQA
- ✅️Text-only QA。比如基于Caption回答问题

# Quickstart
## 启动docker环境

```
docker exec -it qwen_omni bash

cd /workspace/aqa_lalm
```

有时候Docker会掉GPU。可以重启Docker服务：

```
docker stop qwen_omni
docker start qwen_omni
```

## 

nohup python run_eval.py \
  --model QwenOmni \
  --prompt_type caption_only \
  --max_samples 1000

- aqa：直接基于audio问答
- caption_aqa：
- caption_only：基于caption回答问题

## 启动vllm

vllm serve /path/to/Qwen2.5-Omni-7B/ --port 8000 --host 127.0.0.1 --dtype bfloat16

分析结果：
```
python analyze_results.py -i outputs/*.json
```

# TODO
- 输出的结果文件不要用时间戳表示了，太多了，根本不知道语境了。比如改成：模型名_问答方式_results.json

> 一旦进行reasoning，模型的**推理速度会大幅下降**Sd96xnbcvr08。我们需要一些优化手段来提升推理效率。
```
1️⃣ vLLM / batch推理（提速10倍）

2️⃣ 多卡并行（3090全吃满）

```

# Benchmark 评测设计思路

提供选项ABCD。LLM prompt要求直接给出A/B/C/D的答案。


