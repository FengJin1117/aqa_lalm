from models.qwen2_audio import Qwen2AudioAQA

# 使用./models/qwen2_audio.py中的Qwen2AudioModel进行测试。测试样例在./examples/example1.wav，问题是“这个音频讲了什么”
def test_qwen2_audio_infer():
    # Initialize the model
    model = Qwen2AudioAQA()

    # Define the test audio file, question, and choices
    audio_file = "./examples/example1.wav"
    question = "What is the sound in the audio?"
    choices = ["Dog", "Cat", "Car", "Rain"]

    # Run the model inference
    result = model.infer(audio_file, question, choices, test_mode=True)

    # Print the result for verification
    print("Test Result:", result)

    # Assert the result is not empty
    assert result in ["A", "B", "C", "D"], "Result should be one of A, B, C, or D"

if __name__ == "__main__":
    test_qwen2_audio_infer()

