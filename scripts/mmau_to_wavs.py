import os
import json
import io
import soundfile as sf
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

SAVE_DIR = "/data/fengwenhao/datasets/test-mini-audios"
TARGET_SR = 16000

os.makedirs(SAVE_DIR, exist_ok=True)


def extract_id(sample):
    try:
        attr = json.loads(sample["other_attributes"])
        return attr.get("id")
    except:
        return None


def decode_audio_from_bytes(audio_bytes):
    """
    ✅ 用 soundfile 从 bytes 解码（绕开 torchcodec）
    """
    with io.BytesIO(audio_bytes) as f:
        audio_array, sr = sf.read(f)

    # 转 mono
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)

    # 重采样
    if sr != TARGET_SR:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    return audio_array, sr


def main():
    dataset = load_dataset(
        "gamma-lab-umd/MMAU-test-mini",
        split="test",
        streaming=False  # 用本地 cache
    )

    for sample in tqdm(dataset):
        context = sample["context"]

        if context is None:
            continue

        audio_bytes = context.get("bytes", None)

        if audio_bytes is None:
            continue

        # ✅ 手动 decode
        audio_array, sr = decode_audio_from_bytes(audio_bytes)

        uid = extract_id(sample)
        if uid is None:
            continue

        save_path = os.path.join(SAVE_DIR, f"{uid}.wav")

        sf.write(save_path, audio_array, sr, subtype="PCM_16")

    print("✅ Done")


if __name__ == "__main__":
    main()