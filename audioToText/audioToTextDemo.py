import whisper
import os

# Loads local speech AI model
model = whisper.load_model("base")

# Audio file path
audio_path = r"data/audio.mp3"

# Check if file exists
if not os.path.exists(audio_path):
    print(f"File not found: {audio_path}")
else:
    try:
        # Transcribe audio
        result = model.transcribe(audio_path, fp16=False)

        # Print text
        print("Transcribed Text:")
        print(result["text"])

    except Exception as e:
        print(f"Error: {e}")