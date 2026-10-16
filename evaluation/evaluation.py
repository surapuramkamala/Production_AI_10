
import json
from pathlib import Path

from rag_engine import EnterpriseRAG


DATASET_PATH = Path("evaluation/evaluation_dataset.json")
RESULTS_PATH = Path("evaluation/evaluation_results.json")


def load_questions(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))

    # Support either a JSON array or {"questions": [...]}.
    if isinstance(data, dict):
        data = data.get("questions", [])

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must contain a list of questions.")

    return data


def evaluate():
    rag = EnterpriseRAG()

    if rag.index is None:
        raise RuntimeError(
            "No FAISS index found. Start the app and reindex documents first."
        )

    dataset = load_questions(DATASET_PATH)
    results = []

    for item in dataset:
        question = item.get("question", "")
        if not question:
            continue

        expected_sources = item.get(
            "expected_sources",
            item.get("expected_source_filenames", []),
        )
        expected_keywords = item.get("expected_keywords", [])

        if isinstance(expected_sources, str):
            expected_sources = [expected_sources]
        if isinstance(expected_keywords, str):
            expected_keywords = [expected_keywords]

        record = {
            "question": question,
            "stages": {},
        }

        for stage in (
            "basic", "hybrid", "reranking", "query_transformation"
        ):
            output = rag.answer_question(question, stage=stage)

            answer = output["answer"].lower()
            retrieved_sources = {
                source["document"].lower()
                for source in output["sources"]
            }

            source_matches = [
                str(source).lower() in retrieved_sources
                for source in expected_sources
            ]

            keyword_matches = [
                str(keyword).lower() in answer
                for keyword in expected_keywords
            ]

            record["stages"][stage] = {
                "answer": output["answer"],
                "sources": output["sources"],
                "expected_source_match_rate": (
                    sum(source_matches) / len(source_matches)
                    if source_matches else None
                ),
                "expected_keyword_match_rate": (
                    sum(keyword_matches) / len(keyword_matches)
                    if keyword_matches else None
                ),
            }

        results.append(record)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Evaluated {len(results)} questions.")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()
