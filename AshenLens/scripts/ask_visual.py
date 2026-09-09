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

from backend.app.vision.analyzer import (
    VisionAnalyzer,
)


VISUAL_INDEX_FILE = (
    ROOT_DIR
    / "storage"
    / "visual_assets.jsonl"
)


MIN_VISUAL_SCORE = 60.0


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Retrieve visual evidence "
            "and answer a question "
            "using a vision model."
        )
    )

    parser.add_argument(
        "question",
        type=str,
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # 1. Retrieve visual
    # -----------------------------------------------------

    retriever = VisualRetriever(
        VISUAL_INDEX_FILE
    )

    results = retriever.search(
        question=args.question,
        top_k=1,
    )

    if not results:

        print(
            "No visual evidence found."
        )
        return

    result = results[0]

    asset = result.asset

    # -----------------------------------------------------
    # 2. Confidence guard
    # -----------------------------------------------------

    if result.score < MIN_VISUAL_SCORE:

        print(
            "No sufficiently confident "
            "visual match was found."
        )

        print(
            f"Best score: "
            f"{result.score:.2f}"
        )

        return

    # -----------------------------------------------------
    # 3. Print retrieved evidence
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("AshenLens - Visual Question Answering")
    print("=" * 72)

    print()
    print(
        f"Question : {args.question}"
    )

    print()
    print("Retrieved visual")
    print("-" * 72)

    print(
        f"Entity   : "
        f"{asset.get('entity_name')}"
    )

    print(
        f"Type     : "
        f"{asset.get('asset_type')}"
    )

    print(
        f"Score    : "
        f"{result.score:.2f}"
    )

    print(
        f"Source   : "
        f"{asset.get('source_path')}"
    )

    # -----------------------------------------------------
    # 4. Vision analysis
    # -----------------------------------------------------

    print()
    print(
        "Analyzing image..."
    )

    analyzer = VisionAnalyzer()

    answer = (
        analyzer.answer_question(
            question=args.question,

            image_path=Path(
                asset["absolute_path"]
            ),

            entity_name=asset.get(
                "entity_name"
            ),

            asset_type=asset.get(
                "asset_type"
            ),
        )
    )

    # -----------------------------------------------------
    # 5. Answer
    # -----------------------------------------------------

    print()
    print("Answer")
    print("-" * 72)

    print(answer)

    print()
    print("Evidence")
    print("-" * 72)

    print(
        asset.get(
            "source_path"
        )
    )

    if asset.get(
        "related_document"
    ):

        print(
            "Related document: "
            + asset[
                "related_document"
            ]
        )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()