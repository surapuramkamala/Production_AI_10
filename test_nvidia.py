
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: OPENAI_API_KEY is missing from .env")
    raise SystemExit(1)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key='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',
    timeout=30.0,
    max_retries=0,
)

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": "Reply with OK"}
        ],
        max_tokens=10,
    )
    print("NVIDIA connection successful!")
    print(response.choices[0].message.content)

except Exception as exc:
    print("NVIDIA connection failed")
    print("Error type:", type(exc).__name__)
    print("Error:", repr(exc))
