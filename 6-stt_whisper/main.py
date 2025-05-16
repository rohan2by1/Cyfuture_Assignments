from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device="cpu"
)

mp3_file = "test.mp3"  # Replace with your filename
result = pipe(mp3_file)
print(result["text"])