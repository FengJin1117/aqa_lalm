# LALM AQA
这是一个统一 **LALM AQA推理封装**项目。**支持各类Audio QA Benchmark的评测**。
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

## Caption 推理

模型相关参数放在 `conf/{caption_model}.yaml`，目前包含：

- `hf_tag`
- `max_new_tokens`
- `prompt`

推理入口只保留数据输入、输出位置和选择哪个 caption 模型。生成时会一边 caption 一边写入 jsonl，格式为 `{"id": 音频文件名去后缀, "caption": caption文本}`。


“两种 benchmark 回答模式”：aqa 和 caption_qa
- aqa：直接基于音频问答
- caption_qa：基于caption文本问答

```bash
# 基于音频回答
python run_eval.py --benchmark_mode aqa --audio_model QwenOmni --max_samples 1000

# 基于 caption 回答
python run_eval.py \
  --benchmark_mode caption_qa \
  --caption_path data/mmau-test-mini-captions.jsonl \
  --max_samples 1000

# 可以指定vllm服务布置在哪个gpu
python run_eval.py \
  --benchmark_mode caption_qa \
  --caption_path /data2/fwh/audio_caption/outputs/captions_empty.jsonl \
  --vllm_gpu 6 \
  --max_samples 1000 > logs/cap:empty_ans:wen3-4B-instruct_caption_only.log 2>&1 & 
```




如需使用 Qwen2.5-Omni 生成 caption，把 `--caption_model` 改成 `qwen25_omni` 即可，对应配置文件为 `conf/qwen25_omni.yaml`。

## MMAU 评测

```bash
nohup python run_eval.py \
  --task eval \
  --model QwenOmni \
  --prompt_type caption_only \
  --caption_path ./outputs/qwen3_captioner/mmau-test-mini-captions.jsonl \
  --max_samples 1000
```

- aqa：直接基于audio问答
- caption_aqa：先描述音频，再基于音频推理回答
- caption_only：基于caption回答问题

## 启动vllm

vllm serve /path/to/Qwen2.5-Omni-7B/ --port 8000 --host 127.0.0.1 --dtype bfloat16


## Qwen3-4B-Instruct-2507 vLLM 性能测试

实验目录：`exp/vllm_parallel/`

使用 `conda run -n vllm`，在 GPU 4-7 上测试 `Qwen/Qwen3-4B-Instruct-2507`。多卡测试采用多实例部署，每张 GPU 启动一个单卡 vLLM 服务，压测客户端按 round-robin 分发请求。

```bash
# 单卡部署：GPU 4，端口 8000
bash vllm.sh serve-one 4 8000

# 多实例部署：GPU 4/5/6/7 -> 端口 8000/8001/8002/8003
bash vllm.sh serve-many 4

# 串行、并发、多实例扩容测试
bash vllm.sh bench-serial
bash vllm.sh bench-concurrent
bash vllm.sh bench-scale

# 生成测试报告
bash vllm.sh report
```

详细命令和报告说明见 `exp/vllm_parallel/README.md`，报告输出到 `exp/vllm_parallel/report.md`。

分析结果：
```
python analyze_results.py -i outputs/*.json
```

# TODO

- 这里 run_eval.py 的需要合并 "aqa", "caption_aqa"。本质上都是基于audio问答的，只不过一个是直接问答，一个是先生成caption再问答。可以通过参数控制是否生成caption。

```
parser.add_argument(
    "--prompt_type",
    type=str,
    choices=["aqa", "caption_aqa", "caption_only"],
    default="aqa"
)
```

- 修改jsonl格式。

# TODO
- 输出的结果文件不要用时间戳表示了，太多了，根本不知道语境了。比如改成：模型名_问答方式_results.json

> 一旦进行reasoning，模型的**推理速度会大幅下降**Sd96xnbcvr08。我们需要一些优化手段来提升推理效率。
```
1️⃣ vLLM / batch推理（提速10倍）

2️⃣ 多卡并行（3090全吃满）

```

# Benchmark 评测设计思路

提供选项ABCD。LLM prompt要求直接给出A/B/C/D的答案。


