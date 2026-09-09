import argparse
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


VISUAL_INDEX_FILE = (
    ROOT_DIR
    / "storage"
    / "visual_assets.jsonl"
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Search AshenLens visual assets "
            "using a natural-language question."
        )
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question to search visuals for",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return",
    )

    args = parser.parse_args()

    retriever = VisualRetriever(
        VISUAL_INDEX_FILE
    )

    results = retriever.search(
        question=args.question,
        top_k=args.top_k,
    )

    print()
    print("=" * 72)
    print("AshenLens - Visual Search")
    print("=" * 72)

    print()
    print(
        f"Question: {args.question}"
    )

    print()
    print(
        f"Results: {len(results)}"
    )

    print()

    for index, result in enumerate(
        results,
        start=1,
    ):

        asset = result.asset

        print("-" * 72)

        print(
            f"#{index}"
        )

        print(
            f"Score       : "
            f"{result.score:.2f}"
        )

        print(
            f"Entity      : "
            f"{asset.get('entity_name')}"
        )

        print(
            f"Type        : "
            f"{asset.get('asset_type')}"
        )

        print(
            f"Category    : "
            f"{asset.get('category')}"
        )

        print(
            f"File        : "
            f"{asset.get('source_path')}"
        )

        print(
            f"Related doc : "
            f"{asset.get('related_document')}"
        )

        if result.reasons:

            print(
                "Reasons     : "
                + "; ".join(
                    result.reasons
                )
            )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()