# Qwen3-4B-Instruct-2507 vLLM 部署性能测试

实验名：Qwen3-4B-Instruct-2507的vllm部署性能测试

本实验使用 `conda run -n vllm`，在 GPU 4-7 上测试 `Qwen/Qwen3-4B-Instruct-2507` 的 vLLM OpenAI-compatible 服务性能。多卡测试采用多实例部署：每张 GPU 启动一个单卡 vLLM 服务，压测客户端按 round-robin 分发请求。

## 服务启动

单卡部署，GPU 4，端口 8000：

```bash
bash vllm.sh serve-one 4 8000
```

等价展开命令：

```bash
CUDA_VISIBLE_DEVICES=4 VLLM_USE_MODELSCOPE=true \
conda run -n vllm vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 32768 \
  --served-model-name Qwen3-4B-Instruct-2507 \
  --disable-log-requests
```

多实例部署，GPU 4/5/6/7 分别对应端口 8000/8001/8002/8003：

```bash
bash vllm.sh serve-many 4
```

## 串行测试

单卡服务，测单个问题的平均回复时间：

```bash
bash vllm.sh bench-serial
```

展开命令：

```bash
conda run -n vllm python exp/vllm_parallel/bench_chat.py \
  --base-urls http://127.0.0.1:8000/v1 \
  --concurrency 1 \
  --num-requests 20 \
  --name serial_single_gpu
```

## 并发测试

单卡服务，分别测试并发 1、2、4：

```bash
bash vllm.sh bench-concurrent
```

也可以单独运行并发 4：

```bash
conda run -n vllm python exp/vllm_parallel/bench_chat.py \
  --base-urls http://127.0.0.1:8000/v1 \
  --concurrency 4 \
  --num-requests 40 \
  --name concurrent_c4_single_gpu
```

## 多卡多实例测试

先启动 1-4 个实例，再运行扩容测试。`bench-scale` 会依次测试：

- 1 卡：`http://127.0.0.1:8000/v1`
- 2 卡：`http://127.0.0.1:8000/v1`、`http://127.0.0.1:8001/v1`
- 3 卡：端口 8000、8001、8002
- 4 卡：端口 8000、8001、8002、8003

```bash
bash vllm.sh serve-many 4
bash vllm.sh bench-scale
```

展开的 4 实例压测命令：

```bash
conda run -n vllm python exp/vllm_parallel/bench_chat.py \
  --base-urls \
    http://127.0.0.1:8000/v1 \
    http://127.0.0.1:8001/v1 \
    http://127.0.0.1:8002/v1 \
    http://127.0.0.1:8003/v1 \
  --concurrency 4 \
  --num-requests 80 \
  --name scale_4gpu
```

## 输出与报告

压测脚本会写入：

- 明细：`exp/vllm_parallel/results/<name>.jsonl`
- 汇总：`exp/vllm_parallel/results/<name>_summary.json`

生成 Markdown 报告：

```bash
bash vllm.sh report
```

报告路径：

```text
exp/vllm_parallel/report.md
```

脚本会统计成功数、失败数、成功率、平均延迟、p50/p95/p99、吞吐 req/s，并在多实例表格里计算相对单实例加速比。

## Smoke Test

单卡服务启动后，可以先运行小样本测试：

```bash
conda run -n vllm python exp/vllm_parallel/bench_chat.py \
  --base-urls http://127.0.0.1:8000/v1 \
  --concurrency 1 \
  --num-requests 2 \
  --max-tokens 32 \
  --name smoke_single_gpu
```
