from models.qwen_omni import QwenOmni
from prompts.aqa_prompt import AQAPrompt
from prompts.caption_aqa_prompt import CaptionAQAPrompt


def test_aqa():
    model = QwenOmni()
    prompt = AQAPrompt()

    conversation = prompt.build(
        question="How many people are taking part in the talk??",
        choices=["two", "four", "one", "three"],
        audio_path="./examples/3bb27627-0763-4a7f-99c7-5e4c88f85979.wav"
    )

    output = model.infer(conversation, debug=True)
    print("AQA Output:", output)


def test_caption_aqa():
    model = QwenOmni()
    prompt = CaptionAQAPrompt()

    conversation = prompt.build(
        question="How many people are taking part in the talk??",
        choices=["two", "four", "one", "three"],
        audio_path="./examples/3bb27627-0763-4a7f-99c7-5e4c88f85979.wav"
    )

    output = model.infer(conversation, debug=True)
    print("Caption-AQA Output:", output)


if __name__ == "__main__":
    # print("==== TEST AQA ====")
    # test_aqa()

    print("\n==== TEST CAPTION AQA ====")
    test_caption_aqa()