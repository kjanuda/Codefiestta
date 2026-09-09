import json
import sys

from pathlib import Path


ROOT_DIR = Path(
    __file__
).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


from backend.app.retrieval.visual_retriever import (
    VisualRetriever,
)


CORPUS_DIR = (
    ROOT_DIR
    / "corpus"
    / "Ashen_Era_Archive"
)

QUESTIONS_FILE = (
    CORPUS_DIR
    / "sample_questions.json"
)

VISUAL_INDEX_FILE = (
    ROOT_DIR
    / "storage"
    / "visual_assets.jsonl"
)


def main():

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    questions_1a = [
        question
        for question in questions
        if str(
            question.get("track", "")
        ).startswith("1A")
    ]

    retriever = VisualRetriever(
        VISUAL_INDEX_FILE
    )

    print("=" * 80)
    print(
        "AshenLens - 1A Visual Retrieval Test"
    )
    print("=" * 80)

    for question in questions_1a:

        results = retriever.search(
            question=question["question"],
            top_k=3,
        )

        print()
        print(
            f"[{question['qid']}]"
        )

        print(
            question["question"]
        )

        if not results:

            print(
                "  NO RESULTS"
            )

            continue

        for rank, result in enumerate(
            results,
            start=1,
        ):

            asset = result.asset

            print(
                f"  #{rank} "
                f"{result.score:.2f} | "
                f"{asset.get('entity_name')} | "
                f"{asset.get('asset_type')} | "
                f"{asset.get('source_path')}"
            )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()