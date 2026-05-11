from .base_prompt import BasePrompt


class CaptionOnlyPrompt(BasePrompt):
    """
    纯 Caption QA（完全禁止使用音频）
    """

    SYS_PROMPT = (
        "You are an expert in reasoning over audio descriptions.\n"
        "You MUST answer the question ONLY based on the given caption.\n"
        "Do NOT assume any information not explicitly stated in the caption.\n"
        "Choose the best answer from the options.\n"
        "Only output the final answer text."
    )

    def build(self, question, choices, caption, **kwargs):
        choice_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])

        user_prompt = (
            f"Caption:\n{caption}\n\n"
            f"Question:\n{question}\n\n"
            f"Choices:\n{choice_str}\n\n"
            "Answer:"
        )

        # return [
        #     {"role": "system", "content": self.SYS_PROMPT},
        #     {"role": "user", "content": user_prompt}
        # ]

        # 注意：适配多模态模型的输入格式
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.SYS_PROMPT}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]