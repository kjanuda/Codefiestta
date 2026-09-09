import argparse
import json
import sys
import time

from datetime import datetime, timezone
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

EVALUATION_DIR = (
    ROOT_DIR
    / "evaluation"
)

RESULTS_FILE = (
    EVALUATION_DIR
    / "vision_results_1a.json"
)


MIN_VISUAL_SCORE = 60.0


def load_questions() -> list[dict]:

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        questions = json.load(file)

    return [
        question
        for question in questions
        if str(
            question.get(
                "track",
                "",
            )
        ).startswith("1A")
    ]


def load_existing_results() -> dict:

    if not RESULTS_FILE.exists():
        return {}

    try:

        data = json.loads(
            RESULTS_FILE.read_text(
                encoding="utf-8"
            )
        )

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}

    existing = {}

    for item in data.get(
        "results",
        [],
    ):
        qid = item.get("qid")

        if qid:
            existing[qid] = item

    return existing


def save_results(
    results: list[dict],
):

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful = sum(
        1
        for result in results
        if result.get("status")
        == "success"
    )

    failed = sum(
        1
        for result in results
        if result.get("status")
        == "error"
    )

    output = {
        "evaluation": (
            "Sub-track 1A Vision QA"
        ),

        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "question_count": len(
            results
        ),

        "successful": successful,

        "failed": failed,

        "results": results,
    }

    RESULTS_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AshenLens vision "
            "question answering on the "
            "1A development questions."
        )
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help=(
            "Seconds to wait between "
            "API requests."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of questions "
            "to run."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore previously saved "
            "successful results."
        ),
    )

    args = parser.parse_args()

    questions = load_questions()

    if args.limit is not None:

        questions = questions[
            :args.limit
        ]

    existing = (
        {}
        if args.fresh
        else load_existing_results()
    )

    retriever = VisualRetriever(
        VISUAL_INDEX_FILE
    )

    analyzer = VisionAnalyzer()

    results = []

    print("=" * 80)
    print(
        "AshenLens - 1A Vision QA Evaluation"
    )
    print("=" * 80)

    for index, question_data in enumerate(
        questions,
        start=1,
    ):

        qid = question_data["qid"]
        question = question_data[
            "question"
        ]

        print()
        print(
            f"[{index:02}/{len(questions):02}] "
            f"{qid}"
        )

        print(question)

        # ---------------------------------------------
        # Resume successful previous result
        # ---------------------------------------------

        if (
            qid in existing
            and existing[qid].get(
                "status"
            ) == "success"
            and not args.fresh
        ):

            print(
                "Using saved result."
            )

            results.append(
                existing[qid]
            )

            continue

        started = time.perf_counter()

        try:

            visual_results = (
                retriever.search(
                    question=question,
                    top_k=1,
                )
            )

            if not visual_results:

                raise RuntimeError(
                    "No visual result found."
                )

            visual_result = (
                visual_results[0]
            )

            asset = (
                visual_result.asset
            )

            if (
                visual_result.score
                < MIN_VISUAL_SCORE
            ):

                raise RuntimeError(
                    "Visual retrieval score "
                    "below confidence threshold: "
                    f"{visual_result.score:.2f}"
                )

            print(
                "Visual : "
                f"{asset.get('source_path')}"
            )

            print(
                "Score  : "
                f"{visual_result.score:.2f}"
            )

            answer = (
                analyzer.answer_question(
                    question=question,

                    image_path=Path(
                        asset[
                            "absolute_path"
                        ]
                    ),

                    entity_name=asset.get(
                        "entity_name"
                    ),

                    asset_type=asset.get(
                        "asset_type"
                    ),
                )
            )

            elapsed_ms = round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                2,
            )

            result = {
                "qid": qid,

                "question": question,

                "status": "success",

                "answer": answer,

                "visual_score": round(
                    visual_result.score,
                    2,
                ),

                "entity_name":
                    asset.get(
                        "entity_name"
                    ),

                "asset_type":
                    asset.get(
                        "asset_type"
                    ),

                "visual_source":
                    asset.get(
                        "source_path"
                    ),

                "related_document":
                    asset.get(
                        "related_document"
                    ),

                "latency_ms":
                    elapsed_ms,
            }

            print(
                f"Answer : {answer}"
            )

            print(
                f"Time   : "
                f"{elapsed_ms} ms"
            )

        except Exception as error:

            elapsed_ms = round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                2,
            )

            result = {
                "qid": qid,

                "question": question,

                "status": "error",

                "error": str(error),

                "latency_ms":
                    elapsed_ms,
            }

            print(
                f"ERROR  : {error}"
            )

        results.append(
            result
        )

        # Save after every question.
        # If the process stops, progress
        # is not lost.
        save_results(
            results
        )

        if (
            index < len(questions)
            and args.delay > 0
        ):

            print(
                f"Waiting "
                f"{args.delay}s..."
            )

            time.sleep(
                args.delay
            )

    save_results(
        results
    )

    successful = sum(
        1
        for result in results
        if result["status"]
        == "success"
    )

    print()
    print("=" * 80)

    print(
        f"Completed : "
        f"{len(results)}"
    )

    print(
        f"Successful: "
        f"{successful}"
    )

    print(
        f"Failed    : "
        f"{len(results) - successful}"
    )

    print()
    print(
        f"Results:\n"
        f"{RESULTS_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()