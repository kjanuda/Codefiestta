import argparse
import json
import sys

from pathlib import Path


ROOT_DIR = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT_DIR),
)


from backend.app.engine.rich_answer_engine import (
    RichAnswerEngine,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question",
        type=str,
    )

    args = parser.parse_args()

    print("=" * 78)
    print(
        "AshenLens - Rich "
        "Multimodal Answer"
    )
    print("=" * 78)

    print()

    engine = RichAnswerEngine()

    result = engine.answer(
        args.question
    )

    print(
        result["response"]
    )

    print()
    print("-" * 78)

    print(
        "Model:",
        result["model"],
    )

    print(
        "Visual used:",
        result["visual"]["used"],
    )

    print(
        "Visual source:",
        result["visual"][
            "source_path"
        ],
    )

    print()

    print("Text sources:")

    for source in (
        result["text_sources"]
    ):

        print(
            "-",
            source["source_path"],
            "| score:",
            source["score"],
        )


if __name__ == "__main__":
    main()