import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


class QwenOmniAQA:
    def __init__(self, model_name="Qwen/Qwen2.5-Omni-7B"):
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,   # ✅ bf16
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

        # ✅ 关键优化
        self.model.disable_talker()

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

        # ✅ AQA专用 system prompt（比官方更task-specific）
        self.sys_prompt = (
            "You are an audio question answering system. "
            "Answer the question based only on the given audio. "
            "Keep the answer concise and accurate."
        )

    def build_prompt(self, question, choices):
        option_str = "\n".join(
            [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
        )

        few_shot = (
            "Example:\n"
            "Question: What is the sound?\n"
            "A. Dog\nB. Cat\nC. Car\nD. Rain\n"
            "Answer: A\n\n"
        )

        prompt = (
            f"{few_shot}"
            f"{question}\n"
            f"{option_str}\n"
            "Select the correct answer. Respond with only A, B, C, or D."
        )

        return prompt

    def infer(self, audio_path, question, choices, max_new_tokens=128, test_mode=False):
        question = self.build_prompt(question, choices)

        if test_mode:
            print("==== DEBUG PROMPT ====")
            print(question)
            print("======================")

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.sys_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": question},
                ],
            },
        ]

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
            thinker_do_sample=False,   # ✅ 关键
            thinker_max_new_tokens=max_new_tokens
        )

        output = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output