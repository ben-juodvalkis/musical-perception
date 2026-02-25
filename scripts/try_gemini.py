"""Quick experiment: send video to Gemini and see what it understands."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY not set in .env")
    sys.exit(1)

client = genai.Client(api_key=api_key)

video_path = Path(__file__).resolve().parents[1] / "video" / "IMG_7843.MOV"
print(f"Uploading {video_path.name} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)...")

video_file = client.files.upload(file=video_path)
print(f"Uploaded: {video_file.name}, state={video_file.state}")

# Wait for processing
import time
while video_file.state.name == "PROCESSING":
    print("  waiting for processing...")
    time.sleep(2)
    video_file = client.files.get(name=video_file.name)

if video_file.state.name == "FAILED":
    print(f"Upload failed: {video_file.state}")
    sys.exit(1)

print(f"Ready: {video_file.state}")
print()

# Ask Gemini to describe what it sees and hears
prompt = """You are analyzing a video of a dance teacher. Describe:

1. What is happening in the video?
2. Can you hear counting or speech? If so, what words?
3. What type of dance exercise is being demonstrated (if identifiable)?
4. What is the approximate tempo or speed of the movement/counting?
5. Any other observations about rhythm, emphasis, or structure?

Be specific and detailed."""

print("Sending to Gemini 2.5 Flash...")
print("=" * 60)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=video_file.uri,
                    mime_type=video_file.mime_type,
                ),
                types.Part.from_text(text=prompt),
            ],
        )
    ],
)

print(response.text)
print("=" * 60)
print(f"\nUsage: {response.usage_metadata}")
