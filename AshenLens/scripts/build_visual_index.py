import json
import sys

from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


from backend.app.ingestion.corpus_scanner import (
    scan_corpus,
)

from backend.app.ingestion.image_indexer import (
    build_visual_asset,
)


CORPUS_DIR = (
    ROOT_DIR
    / "corpus"
    / "Ashen_Era_Archive"
)

STORAGE_DIR = (
    ROOT_DIR
    / "storage"
)

VISUAL_INDEX_FILE = (
    STORAGE_DIR
    / "visual_assets.jsonl"
)

VISUAL_SUMMARY_FILE = (
    STORAGE_DIR
    / "visual_summary.json"
)


def main():

    print("=" * 70)
    print("AshenLens - Visual Asset Index Builder")
    print("=" * 70)

    STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus_files = scan_corpus(
        CORPUS_DIR
    )

    image_files = [
        file
        for file in corpus_files
        if file.kind == "image"
    ]

    print()
    print(
        f"Discovered images : {len(image_files)}"
    )

    print()
    print("Indexing visual assets...")
    print("-" * 70)

    assets = []

    errors = []

    for index, corpus_file in enumerate(
        image_files,
        start=1,
    ):

        print(
            f"[{index:03}/{len(image_files):03}] "
            f"{corpus_file.source_path}"
        )

        try:

            asset = build_visual_asset(
                corpus_file=corpus_file,
                corpus_dir=CORPUS_DIR,
            )

            assets.append(asset)

        except Exception as error:

            print(
                f"    ERROR: {error}"
            )

            errors.append(
                {
                    "source_path":
                        corpus_file.source_path,

                    "error":
                        str(error),
                }
            )

    # -----------------------------------------------------
    # Write JSONL index
    # -----------------------------------------------------

    with VISUAL_INDEX_FILE.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for asset in assets:

            output_file.write(
                asset.model_dump_json()
            )

            output_file.write("\n")

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    type_counts = Counter(
        asset.asset_type
        for asset in assets
    )

    category_counts = Counter(
        asset.category
        for asset in assets
    )

    related_document_count = sum(
        1
        for asset in assets
        if asset.related_document
    )

    summary = {
        "discovered_images":
            len(image_files),

        "indexed_assets":
            len(assets),

        "assets_with_related_document":
            related_document_count,

        "assets_by_type":
            dict(
                sorted(
                    type_counts.items()
                )
            ),

        "assets_by_category":
            dict(
                sorted(
                    category_counts.items()
                )
            ),

        "errors":
            errors,
    }

    VISUAL_SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)

    print(
        f"Indexed assets    : {len(assets)}"
    )

    print(
        "Related documents : "
        f"{related_document_count}"
    )

    print(
        f"Errors            : {len(errors)}"
    )

    print()
    print(
        f"Visual index:\n{VISUAL_INDEX_FILE}"
    )

    print()
    print(
        f"Summary:\n{VISUAL_SUMMARY_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()