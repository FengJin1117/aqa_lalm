# 项目结构
```
mmau_eval/
├── models/
│   └── qwen_omni.py        # 模型封装（AQA接口）
├── eval/
│   └── mmau_eval.py        # benchmark逻辑
├── data/
│   └── mmau/               # 数据
├── run_eval.py             # 入口
└── README.md
```

# 环境安装

建一个常用环境
```
conda create -n llm python=3.10 -y
conda activate llm

pip install datasets
pip install jupyter

pip install torchcodec

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

# 一些经验
## talker模块需要fp32，但thiker支持fp16 / bp16

flash-attn仅支持精度fp16 / bp16，所以无法为talker加速。
但是能加速LLM（thinker）部分。

无视报错：`Qwen2_5OmniToken2WavModel must inference with fp32`


# 
查看HF缓存：
ls ~/.cache/huggingface/hub/