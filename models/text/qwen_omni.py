import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


class QwenOmni:
    """
    ✅ 只做一件事：
    multimodal input -> model output

    不关心：
    - prompt怎么写
    - few-shot
    - task类型
    """

    def __init__(self, model_name="Qwen/Qwen2.5-Omni-7B"):
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

        self.model.disable_talker()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    def infer(
        self,
        conversation,   # ✅ 完全开放
        max_new_tokens=256,
        debug=False
    ):
        """
        conversation: 标准Qwen chat格式
        """

        if debug:
            print("==== RAW CONVERSATION ====")
            print(conversation)

        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        audios, images, videos = process_mm_info(
            conversation,
            use_audio_in_video=False
        )

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        ).to(self.model.device).to(self.model.dtype)

        output_ids = self.model.generate(
            **inputs,
            return_audio=False,
            thinker_do_sample=False,
            thinker_max_new_tokens=max_new_tokens
        )

        output = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output