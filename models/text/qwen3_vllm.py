import json
import urllib.error
import urllib.request


class Qwen3VLLMText:
    def __init__(
        self,
        base_url="http://127.0.0.1:8000/v1",
        model_name="Qwen3-4B-Instruct-2507",
        temperature=0.0,
        max_tokens=256,
        timeout=120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def infer(self, conversation, debug=False):
        messages = self._conversation_to_messages(conversation)

        if debug:
            print("==== QWEN3 VLLM MESSAGES ====")
            print(messages)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = self._post_json(f"{self.base_url}/chat/completions", payload)
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        return message.get("content", "").strip()

    def _conversation_to_messages(self, conversation):
        messages = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append(
                {
                    "role": role,
                    "content": self._content_to_text(content),
                }
            )
        return messages

    def _content_to_text(self, content):
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts)

        return str(content)

    def _post_json(self, url, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request failed: HTTP {exc.code}: {body}") from exc
