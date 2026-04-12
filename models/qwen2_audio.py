import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa


class Qwen2AudioAQA:
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct"):
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

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
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": question},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        audio, sr = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)

        inputs = self.processor(
            text=text,
            audios=[audio],
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Decode only the newly generated tokens
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output
