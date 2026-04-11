import numpy as np
import soundfile as sf

def generate_noise(out_path, duration_sec, sr=16000):
    print(f"🎧 Generating {duration_sec}s noise audio...")
    audio = np.random.randn(sr * duration_sec).astype(np.float32) * 0.01
    sf.write(out_path, audio, sr)
    print("✅ Done.")

if __name__ == "__main__":
    generate_noise("test_1h.wav", 3600)   # 1小时