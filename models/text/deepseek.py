from llm_processor import DeepSeekAgent


class DeepSeekText:

    def __init__(self):

        self.agent = DeepSeekAgent()

    def infer(self, conversation, debug=False):

        # conversation -> 纯文本prompt
        prompt = ""

        for msg in conversation:

            role = msg["role"].upper()

            content = msg["content"]

            # 兼容你的 multimodal prompt 格式
            if isinstance(content, list):

                text_parts = []

                for item in content:

                    if item["type"] == "text":
                        text_parts.append(item["text"])

                content = "\n".join(text_parts)

            prompt += f"{role}:\n{content}\n\n"

        response = self.agent.chat(
            prompt,
            temperature=0.0,  # QA任务建议低温
            max_tokens=256,
        )

        return response.strip()