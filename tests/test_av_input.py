import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import torch

# ====== Load model ======
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype=torch.bfloat16,
    # torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# ✅ 关键：关闭语音输出（省显存+加速）
# 不要talker，节省2GB显存
model.disable_talker()

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")


# ====== Conversation（核心修改点） ======
conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            }
        ],
    },
    {
        "role": "user",
        "content": [
            # ✅ 音频输入（换成你的路径）
            {"type": "audio", "audio": "output.wav"},

            # ✅ 文本输入（让模型描述音频）
            {"type": "text", "text": "Describe the audio in detail."},
        ],
    },
]

# 👉 这里不涉及video，所以可以关掉
USE_AUDIO_IN_VIDEO = False


# ====== Prepare inputs ======
text = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=False
)

audios, images, videos = process_mm_info(
    conversation,
    use_audio_in_video=USE_AUDIO_IN_VIDEO
)

inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO
)

inputs = inputs.to(model.device).to(model.dtype)


# ====== Inference（只要text） ======
text_ids = model.generate(
    **inputs,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
    return_audio=False,   # ✅ 关键：关闭音频输出
    thinker_do_sample=False
)

text = processor.batch_decode(
    text_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(text)