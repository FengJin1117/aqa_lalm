import os
from openai import OpenAI

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

'''
推理参数设置：
场景	温度
代码生成/数学解题	0.0
数据抽取/分析	    1.0
通用对话	        1.3
翻译	            1.3
创意类写作/诗歌创作	1.5

'''

class DeepSeekAgent:

    def __init__(
        self,
        api_key=None,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    ):

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

        if self.api_key is None:
            raise ValueError("DEEPSEEK_API_KEY not found")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )

        self.model = model

    def chat(
        self,
        prompt,
        temperature=1.3, # 通用对话设置
        max_tokens=1024,
    ):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

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