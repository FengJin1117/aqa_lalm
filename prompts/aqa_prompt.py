from .base_prompt import BasePrompt


class AQAPrompt(BasePrompt):
    """
    标准 AQA（多选）
    """

    SYS_PROMPT = (
        "You are an audio question answering system. "
        "Answer the question based only on the given audio. "
        "Keep the answer concise and accurate."
    )

    FEW_SHOT = (
        "Example:\n"
        "Question: What is the sound?\n"
        "A. Dog\nB. Cat\nC. Car\nD. Rain\n"
        "Answer: A\n\n"
    )

    def build(self, question, choices, audio_path):
        option_str = "\n".join(
            [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
        )

        user_text = (
            f"{self.FEW_SHOT}"
            f"{question}\n"
            f"{option_str}\n"
            "Select the correct answer. Respond with only A, B, C, or D."
        )

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYS_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        return conversation