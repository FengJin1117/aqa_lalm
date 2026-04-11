# LALM AQA
这是一个统一 **LALM AQA推理封装**项目。
输入格式：audio + question

## 📦支持的模型
- ✅qwen 2.5 omni 7B
- TODO: qwen 2 audio。

# Quickstart
## 启动docker

```
docker exec -it qwen_omni bash

cd /workspace/aqa_lalm
```

## 开发时

## 使用时
- 希望能够代码和用户数据分离。保证库的干净。

# TODO
给一个模板类：

```
1️⃣ vLLM / batch推理（提速10倍）

2️⃣ 多卡并行（3090全吃满）

3️⃣ 错误case dump（分析模型弱点）
```

# Benchmark 评测设计思路

提供选项ABCD。LLM prompt要求直接给出A/B/C/D的答案。


