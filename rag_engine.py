
import csv
import json
import re
from pathlib import Path

import faiss
import numpy as np
from docx import Document
from pypdf import PdfReader

from nvidia_client import embed_texts, generate_answer


DATA_DIR = Path("data")
STORE_DIR = Path("vector_store")
INDEX_PATH = STORE_DIR / "faiss.index"
CHUNKS_PATH = STORE_DIR / "chunks.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def read_document(path: Path) -> list[dict]:
    """Return text pages/sections with basic source metadata."""
    suffix = path.suffix.lower()
    results = []

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                results.append({
                    "text": text,
                    "page": page_number,
                })

    elif suffix == ".docx":
        document = Document(str(path))
        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )
        if text.strip():
            results.append({"text": text, "page": None})

    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            results.append({"text": text, "page": None})

    elif suffix == ".csv":
        with path.open(
            "r", encoding="utf-8-sig", errors="ignore", newline=""
        ) as file:
            text = "\n".join(
                ", ".join(row)
                for row in csv.reader(file)
            )
        if text.strip():
            results.append({"text": text, "page": None})

    return results


def split_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


class EnterpriseRAG:
    def __init__(self):
        self.index = None
        self.chunks = []
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.load_index()

    def load_index(self):
        if INDEX_PATH.exists() and CHUNKS_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            self.chunks = json.loads(
                CHUNKS_PATH.read_text(encoding="utf-8")
            )

    def index_documents(self):
        """Rebuild the index from files in data/."""
        if not DATA_DIR.exists():
            raise FileNotFoundError("Create the data/ directory first.")

        new_chunks = []

        supported = {".pdf", ".docx", ".txt", ".csv"}
        files = sorted(
            p for p in DATA_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in supported
        )

        if not files:
            raise ValueError("No supported documents were found in data/.")

        for path in files:
            sections = read_document(path)

            for section in sections:
                for chunk in split_text(section["text"]):
                    new_chunks.append({
                        "document": path.name,
                        "source": path.name,
                        "page": section["page"],
                        "file_type": path.suffix.lower(),
                        "text": chunk,
                        "text_preview": chunk[:250],
                    })

        if not new_chunks:
            raise ValueError("Documents were found, but no text was extracted.")

        # Embed in batches to avoid very large API requests.
        all_vectors = []
        batch_size = 16

        for start in range(0, len(new_chunks), batch_size):
            batch = new_chunks[start:start + batch_size]
            vectors = embed_texts([item["text"] for item in batch])
            all_vectors.extend(vectors)

        matrix = np.asarray(all_vectors, dtype="float32")
        if matrix.ndim != 2 or len(matrix) != len(new_chunks):
            raise RuntimeError("Embedding response did not match chunk count.")

        # Normalize vectors so inner product acts like cosine similarity.
        faiss.normalize_L2(matrix)
        new_index = faiss.IndexFlatIP(matrix.shape[1])
        new_index.add(matrix)

        # Save only after all embeddings and indexing succeed.
        faiss.write_index(new_index, str(INDEX_PATH))
        CHUNKS_PATH.write_text(
            json.dumps(new_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.index = new_index
        self.chunks = new_chunks

        return {
            "documents": len(files),
            "chunks": len(new_chunks),
        }

    def _vector_search(self, query: str, top_k: int = 8):
        if self.index is None or not self.chunks:
            raise RuntimeError(
                "No documents are indexed. Run document reindexing first."
            )

        vector = np.asarray(embed_texts([query]), dtype="float32")
        faiss.normalize_L2(vector)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                item = dict(self.chunks[int(idx)])
                item["score"] = float(score)
                results.append(item)

        return results

    @staticmethod
    def _keyword_search(query: str, top_k: int = 8, chunks=None):
        chunks = chunks or []
        terms = set(re.findall(r"\w+", query.lower()))
        results = []

        for item in chunks:
            text_terms = set(
                re.findall(r"\w+", item["text"].lower())
            )
            overlap = len(terms & text_terms) / max(len(terms), 1)

            if overlap > 0:
                result = dict(item)
                result["keyword_score"] = overlap
                results.append(result)

        results.sort(
            key=lambda item: item["keyword_score"],
            reverse=True,
        )
        return results[:top_k]

    def retrieve(self, query: str, stage: str = "hybrid", top_k: int = 5):
        stage = stage.lower()
        vector_results = self._vector_search(query, top_k=12)

        if stage == "basic":
            results = vector_results
        else:
            keyword_results = self._keyword_search(
                query, top_k=12, chunks=self.chunks
            )

            # Merge candidates by their stable document/chunk identity.
            merged = {}

            for item in vector_results:
                key = (item["document"], item["page"], item["text"])
                merged[key] = dict(item)
                merged[key]["vector_score"] = item["score"]
                merged[key]["keyword_score"] = 0.0

            for item in keyword_results:
                key = (item["document"], item["page"], item["text"])
                if key not in merged:
                    merged[key] = dict(item)
                    merged[key]["vector_score"] = 0.0
                merged[key]["keyword_score"] = item["keyword_score"]

            results = list(merged.values())

            for item in results:
                item["score"] = (
                    0.7 * item.get("vector_score", 0.0)
                    + 0.3 * item.get("keyword_score", 0.0)
                )

            results.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

            if stage == "reranking":
                query_terms = set(re.findall(r"\w+", query.lower()))

                def rerank_score(item):
                    words = set(
                        re.findall(r"\w+", item["text"].lower())
                    )
                    overlap = len(query_terms & words) / max(
                        len(query_terms), 1
                    )
                    return 0.5 * item["score"] + 0.5 * overlap

                results.sort(key=rerank_score, reverse=True)

        if stage not in {
            "basic", "hybrid", "reranking", "query_transformation"
        }:
            raise ValueError(f"Unsupported retrieval stage: {stage}")

        return results[:top_k]

    def answer_question(self, question: str, stage: str = "hybrid"):
        # Starter query transformation: normalize whitespace.
        # Replace this with an LLM-based query rewriter when needed.
        rewritten_query = " ".join(question.split())

        chunks = self.retrieve(
            rewritten_query,
            stage=stage,
            top_k=5,
        )

        if not chunks:
            answer = "I could not find relevant information in the indexed documents."
        else:
            context = "\n\n".join(
                f"[Source: {item['document']}, page: {item['page']}]\n"
                f"{item['text']}"
                for item in chunks
            )

            prompt = f"""
Answer the question using only the context below.

Rules:
- Do not make up facts that are absent from the context.
- If the context is insufficient, clearly say so.
- Cite source filenames in your answer.
- Distinguish what the documents state from any uncertainty.

Context:
{context}

Question:
{question}

Answer:
"""
            answer = generate_answer(prompt)

        sources = []
        seen = set()

        for item in chunks:
            key = (item["document"], item["page"], item["text_preview"])
            if key in seen:
                continue
            seen.add(key)

            sources.append({
                "document": item["document"],
                "source": item["source"],
                "page": item["page"],
                "file_type": item["file_type"],
                "score": round(float(item.get("score", 0.0)), 4),
                "text_preview": item["text_preview"],
            })

        return {
            "answer": answer,
            "confidence": (
                round(float(chunks[0].get("score", 0.0)), 4)
                if chunks else 0.0
            ),
            "sources": sources,
            "rewritten_query": rewritten_query,
            "retrieval_stage": stage,
            "retrieved_chunks": chunks,
        }

    # Compatibility with code that calls ask().
    def ask(self, question: str, stage: str = "hybrid"):
        return self.answer_question(question=question, stage=stage)
