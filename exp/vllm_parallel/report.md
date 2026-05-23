# Qwen3-4B-Instruct-2507的vllm部署性能测试

## 实验设置

- 模型：`Qwen/Qwen3-4B-Instruct-2507`
- served model name：`Qwen3-4B-Instruct-2507`
- conda 环境：`vllm`
- GPU：4, 5, 6, 7
- 多卡口径：多实例部署，每张 GPU 一个单卡 vLLM 服务，客户端 round-robin 分发请求。
- 请求接口：OpenAI-compatible `/v1/chat/completions`

## 启动命令

```bash
bash vllm.sh serve-one 4 8000
bash vllm.sh serve-many 4
```

## 测试命令

```bash
bash vllm.sh bench-serial
bash vllm.sh bench-concurrent
bash vllm.sh bench-scale
bash vllm.sh report
```

## 串行测试

| run | instances | concurrency | requests | success | avg latency(s) | p95(s) | throughput(req/s) | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serial_single_gpu | 1 | 1 | 20 | 100.0% | 1.644 | 1.955 | 0.61 | - |

## 并发测试

| run | instances | concurrency | requests | success | avg latency(s) | p95(s) | throughput(req/s) | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| concurrent_c1_single_gpu | 1 | 1 | 40 | 100.0% | 1.592 | 1.922 | 0.63 | - |
| concurrent_c2_single_gpu | 1 | 2 | 40 | 100.0% | 1.609 | 1.940 | 1.22 | - |
| concurrent_c4_single_gpu | 1 | 4 | 40 | 100.0% | 1.632 | 1.982 | 2.39 | - |

## 多实例扩容测试

| run | instances | concurrency | requests | success | avg latency(s) | p95(s) | throughput(req/s) | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scale_1gpu | 1 | 4 | 80 | 100.0% | 1.638 | 1.982 | 2.41 | 1.00x |
| scale_2gpu | 2 | 4 | 80 | 100.0% | 1.624 | 1.941 | 2.44 | 1.01x |
| scale_3gpu | 3 | 4 | 80 | 100.0% | 1.612 | 1.940 | 2.44 | 1.01x |
| scale_4gpu | 4 | 4 | 80 | 100.0% | 1.602 | 1.925 | 2.46 | 1.02x |

## 结论

本次共完成 460 个请求，成功 460 个，整体成功率 100.0%。
串行单卡平均回复时间为 1.644s，p95 为 1.955s。
单卡并发测试中，并发 4 的吞吐最高，为 2.39 req/s。
多实例扩容测试中，最高吞吐来自 `scale_4gpu`，吞吐为 2.46 req/s。
在固定最多 4 并发的条件下，4 实例相对 1 实例吞吐提升约 1.02x，主要改善是平均延迟和 p95 小幅下降；该负载没有把多实例能力充分压满。
