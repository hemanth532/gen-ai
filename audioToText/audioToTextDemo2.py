from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu")

segments, info = model.transcribe("data/audio.mp3")

for segment in segments:
    print(segment.text)