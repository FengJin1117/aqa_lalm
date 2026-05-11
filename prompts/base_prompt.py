class BasePrompt:
    """
    Prompt 抽象基类（无状态 / 无配置）

    设计原则：
    - 不持有 config
    - 不做任何 IO
    - 只负责：输入 → conversation
    """

    def build(self, **kwargs):
        """
        构造 Qwen 标准 conversation

        Returns:
            List[Dict]: conversation
        """
        raise NotImplementedError("Subclasses must implement build()")