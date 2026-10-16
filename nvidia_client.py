
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY", "")
CHAT_MODEL = os.getenv(
    "NVIDIA_CHAT_MODEL",
    "openai/gpt-oss-20b",
)
EMBEDDING_MODEL = os.getenv(
    "NVIDIA_EMBEDDING_MODEL",
    "nvidia/nv-embedqa-e5-v5",
)

if not API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY is missing. Add it to your .env file."
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create embeddings for a batch of text strings."""
    if not texts:
        return []

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )

    # Restore the original input order.
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def generate_answer(prompt: str) -> str:
    """Generate an answer using the configured NVIDIA chat model."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an enterprise document assistant. "
                    "Answer using the supplied document context only. "
                    "Do not invent facts. If the context does not "
                    "contain the answer, say you do not know. "
                    "Mention the source filenames supporting the answer."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
    )

    return response.choices[0].message.content or ""
