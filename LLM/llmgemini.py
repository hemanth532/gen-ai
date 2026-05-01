import os

from google import genai

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Streaming
for chunk in client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="Write a short poem"
):
    print(chunk.text, end="")