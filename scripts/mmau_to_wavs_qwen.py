import os
import json
import io
import soundfile as sf
import numpy as np
from datasets import load_dataset, Features, Value, Sequence
from tqdm import tqdm

SAVE_DIR = "/data/fengwenhao/datasets/test-mini-audios"
TARGET_SR = 16000

os.makedirs(SAVE_DIR, exist_ok=True)

def extract_id(sample):
    try:
        # 此时 other_attributes 应该是原始字符串或字典，取决于数据集原始格式
        attrs = sample["other_attributes"]
        if isinstance(attrs, str):
            attr = json.loads(attrs)
        else:
            attr = attrs
        return attr.get("id")
    except Exception as e:
        # print(f"Error extracting id: {e}")
        return None

def decode_audio_from_bytes(audio_bytes):
    """
    ✅ 用 soundfile 从 bytes 解码
    """
    if not audio_bytes:
        return None, None
        
    try:
        with io.BytesIO(audio_bytes) as f:
            audio_array, sr = sf.read(f)
    except Exception as e:
        print(f"Soundfile read error: {e}")
        return None, None

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
    # 1. 先加载数据集以获取原始 features
    # 注意：这里可能会报错，如果 load_dataset 过程中就触发了解码？
    # 通常 load_dataset 只是下载/读取元数据，迭代时才解码。
    # 但如果 load_dataset 内部有校验，可能需要 streaming=True 再转换，或者直接修改 cache。
    
    # 为了安全，我们使用 streaming=True 加载，然后手动处理，或者：
    # 【关键技巧】使用 ignore_verifications 或直接修改 features
    
    dataset = load_dataset(
        "gamma-lab-umd/MMAU-test-mini",
        split="test",
        streaming=False
    )
    
    # 2. 【核心修改】重新定义 Features，将可能包含音频的列改为 binary
    # 我们需要知道 'context' 列的具体结构。
    # 假设 context 是一个包含 'bytes' 和 'sampling_rate' 的字典结构。
    # 如果 datasets 把它识别为 Audio 类型，我们需要把它改回普通字典或 binary。
    
    original_features = dataset.features
    # print(original_features) # 调试用，查看结构
    
    # 构造新的 features，将 Audio 类型替换为 Value('binary') 或保持原样但禁用解码
    # 由于 context 可能是嵌套结构，直接替换比较麻烦。
    
    # 【更简单的替代方案】：
    # 直接卸载 torchcodec 是最快的。如果不想卸载，请尝试下面的 try-except 补丁
    
    # --- 尝试直接迭代，但捕获特定的 torchcodec 错误？不行，错误发生在 iter 内部 ---
    
    # --- 真正有效的代码级解决方案 ---
    # 既然 arrow 格式取值麻烦，我们换一个思路：
    # 使用 dataset.cast_column 将音频列 cast 为 binary? 
    # 如果 context 是 Audio 类型，cast_column('context', Value('binary')) 可能会失败，因为结构不匹配。
    
    # 让我们回到最原始的报错：torchcodec 加载失败。
    # 只要让 python 找不到 torchcodec 即可。
    
    import sys
    # 临时从 sys.modules 中移除 torchcodec，防止 datasets 导入它
    if 'torchcodec' in sys.modules:
        del sys.modules['torchcodec']
    if 'datasets.features._torchcodec' in sys.modules:
        del sys.modules['datasets.features._torchcodec']
        
    # 阻止后续导入
    class BlockTorchCodec:
        def find_module(self, fullname, path=None):
            if 'torchcodec' in fullname:
                return self
            return None
        def load_module(self, fullname):
            raise ImportError(f"Blocked import of {fullname} to avoid libnvrtc error")
            
    # 注意：这可能会导致 datasets 报错说没有可用的后端，但它通常会回退到 torchaudio 或 soundfile
    # 如果回退成功，你的代码就能跑通。
    
    # 但由于 datasets 2.18+ 硬依赖 torchcodec 用于某些加速，这可能行不通。
    
    # ==========================================
    # ✅ 最终推荐：使用 pyarrow 直接读取，完全绕过 datasets 的 decode_row
    # ==========================================
    
    # 获取底层的 pyarrow table
    pa_table = dataset.data
    
    for i in tqdm(range(len(pa_table))):
        # 从 pyarrow table 直接取数据，不会触发 datasets 的 feature decode
        row = pa_table.slice(i, 1)
        
        # 提取 other_attributes
        # 假设 other_attributes 是 JSON 字符串列
        try:
            other_attr_str = row["other_attributes"][0].as_py()
            if isinstance(other_attr_str, str):
                attr = json.loads(other_attr_str)
            else:
                attr = other_attr_str
            uid = attr.get("id")
        except:
            continue
            
        if uid is None:
            continue
            
        # 提取 context
        # context 在 arrow 里可能是一个 Struct: {'bytes': ..., 'sampling_rate': ...}
        context_struct = row["context"][0]
        
        if context_struct is None:
            continue
            
        # 访问 struct 中的 bytes 字段
        # 注意：pyarrow struct 的访问方式
        try:
            # 检查 context_struct 是否是 pyarrow Scalar
            if hasattr(context_struct, 'as_py'):
                context_dict = context_struct.as_py()
            else:
                context_dict = context_struct
                
            if not context_dict or 'bytes' not in context_dict:
                continue
                
            audio_bytes = context_dict['bytes']
            
        except Exception as e:
            # print(f"Error parsing context: {e}")
            continue

        if audio_bytes is None:
            continue

        # 手动解码
        audio_array, sr = decode_audio_from_bytes(audio_bytes)
        
        if audio_array is None:
            continue
            
        save_path = os.path.join(SAVE_DIR, f"{uid}.wav")
        sf.write(save_path, audio_array, sr, subtype="PCM_16")

    print("✅ Done")

if __name__ == "__main__":
    main()