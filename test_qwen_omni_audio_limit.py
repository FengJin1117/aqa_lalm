import os
import time
import torch
import librosa
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


# =========================
# 🔧 配置
# =========================
MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
GPU_ID = 1
BASE_AUDIO = "test_1h.wav"

TEST_LENGTHS = [60, 300, 600, 1200， 3600]  # 逐步来，别一上来3600


# =========================
# 🧠 显存工具
# =========================
def print_mem(tag):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{tag}] allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")


# =========================
# 🚀 模型封装
# =========================
class QwenOmniRunner:
    def __init__(self):
        print("🚀 Loading model...")

        self.device = f"cuda:{GPU_ID}"
        torch.cuda.set_device(self.device)

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            attn_implementation="flash_attention_2",
        )

        self.model.disable_talker()

        self.processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)

        self.sys_prompt = "You are an audio question answering system."

        print_mem("after load")
        print("✅ Model loaded.\n")

    def infer(self, audio_path, question):
        # ====== 1. 读音频（关键：手动） ======
        wav, sr = sf.read(audio_path)

        print(f"🎧 audio loaded: shape={wav.shape}, sr={sr}")

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.sys_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        # ====== 2. processor（直接喂 waveform） ======
        inputs = self.processor(
            text=text,
            audio=[wav],   # ✅ 强制音频进入模型
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            truncation=False,   # ❗关键：关闭截断
        )

        # debug
        print("inputs keys:", inputs.keys())
        if "audio" in inputs:
            print("audio tensor shape:", inputs["audio"].shape)

        feat = inputs["input_features"]
        print("feature length:", feat.shape[-1])

        inputs = inputs.to(self.device).to(torch.bfloat16)

        # ====== 3. 显存监控 ======
        torch.cuda.reset_peak_memory_stats()
        print_mem("before infer")

        start = time.time()

        outputs = self.model.generate(
            **inputs,
            return_audio=False,
            thinker_do_sample=False,
            max_new_tokens=64,
        )

        end = time.time()

        print_mem("after infer")
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[peak] {peak:.2f} GB")

        text_out = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        return text_out, end - start


# =========================
# 🎧 裁剪音频
# =========================
def cut_audio(in_path, out_path, seconds):
    audio, sr = librosa.load(in_path, sr=None)
    audio = audio[: sr * seconds]
    sf.write(out_path, audio, sr)


# =========================
# 🧪 测试主流程
# =========================
def run_length_test():
    runner = QwenOmniRunner()

    question = "What is happening in this audio?"

    for t in TEST_LENGTHS:
        print(f"\n🧪 Testing {t} seconds")

        out_audio = f"tmp_{t}.wav"

        try:
            cut_audio(BASE_AUDIO, out_audio, t)

            text, latency = runner.infer(out_audio, question)

            print(f"✅ SUCCESS")
            print(f"⏱ Latency: {latency:.2f}s")
            print(f"📝 Output: {text[:100]}")

        except RuntimeError as e:
            print(f"❌ OOM or Runtime Error: {e}")

        except Exception as e:
            print(f"❌ Other Error: {e}")

        finally:
            if os.path.exists(out_audio):
                os.remove(out_audio)


# =========================
# 🏁 主函数
# =========================
if __name__ == "__main__":
    run_length_test()