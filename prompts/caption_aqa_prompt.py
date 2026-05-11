from .base_prompt import BasePrompt


class CaptionAQAPrompt(BasePrompt):
    """
    Caption增强版 AQA（隐式两阶段）
    """

    # SYS_PROMPT = (
    #     "You are an expert in understanding audio and answering questions based on it."
    # )
    SYS_PROMPT = (
        "You are an audio question answering system. "
        "Answer the question based only on the given audio. "
        "Keep the answer concise and accurate."
    )

    def build(self, question, choices, audio_path):
        option_str = "\n".join(
            [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
        )

        user_text = (
            "You must follow the steps strictly.\n\n"

            "Step 1: Describe the audio in detail.\n"
            "Step 2: Reason about the question based on the audio.\n"
            "Step 3: Output the final answer.\n\n"

            "Format:\n"
            "Description: <your description>\n"
            "Reasoning: <your reasoning>\n"
            "Answer: <A/B/C/D>\n\n"

            f"Question: {question}\n"
            f"{option_str}\n"
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