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

from backend.app.ingestion.text_parser import (
    parse_text_file,
)


CORPUS_DIR = (
    ROOT_DIR
    / "corpus"
    / "Ashen_Era_Archive"
)

STORAGE_DIR = ROOT_DIR / "storage"

DOCUMENTS_FILE = (
    STORAGE_DIR
    / "documents.jsonl"
)

SUMMARY_FILE = (
    STORAGE_DIR
    / "ingestion_summary.json"
)


def main():
    print("=" * 70)
    print("AshenLens - Text Corpus Builder")
    print("=" * 70)

    STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Scanning corpus:\n{CORPUS_DIR}"
    )

    corpus_files = scan_corpus(
        CORPUS_DIR
    )

    text_files = [
        file
        for file in corpus_files
        if file.kind == "text"
    ]

    image_files = [
        file
        for file in corpus_files
        if file.kind == "image"
    ]

    other_files = [
        file
        for file in corpus_files
        if file.kind == "other"
    ]

    print()
    print(
        f"Total physical files : {len(corpus_files)}"
    )

    print(
        f"Text source files    : {len(text_files)}"
    )

    print(
        f"Image source files   : {len(image_files)}"
    )

    print(
        f"Other files          : {len(other_files)}"
    )

    records = []

    errors = []

    print()
    print("Parsing text documents...")
    print("-" * 70)

    for index, corpus_file in enumerate(
        text_files,
        start=1,
    ):

        print(
            f"[{index:03}/{len(text_files):03}] "
            f"{corpus_file.source_path}"
        )

        try:

            parsed_records = parse_text_file(
                corpus_file
            )

            records.extend(
                parsed_records
            )

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

    print()
    print("Writing normalized records...")

    with DOCUMENTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for record in records:

            output_file.write(
                record.model_dump_json()
            )

            output_file.write("\n")

    record_extensions = Counter(
        record.extension
        for record in records
    )

    category_counts = Counter(
        record.category
        for record in records
    )

    ocr_records = [
        record
        for record in records
        if record.needs_ocr
    ]

    empty_text_records = [
        record
        for record in records
        if not record.text.strip()
    ]

    total_characters = sum(
        record.char_count
        for record in records
    )

    summary = {
        "physical_files": len(
            corpus_files
        ),

        "text_source_files": len(
            text_files
        ),

        "image_source_files": len(
            image_files
        ),

        "other_files": len(
            other_files
        ),

        "generated_text_records": len(
            records
        ),

        "total_characters": (
            total_characters
        ),

        "needs_ocr_records": len(
            ocr_records
        ),

        "empty_text_records": len(
            empty_text_records
        ),

        "records_by_extension": dict(
            sorted(
                record_extensions.items()
            )
        ),

        "records_by_category": dict(
            sorted(
                category_counts.items()
            )
        ),

        "errors": errors,
    }

    SUMMARY_FILE.write_text(
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
        f"Generated records : {len(records)}"
    )

    print(
        f"Characters        : {total_characters:,}"
    )

    print(
        f"Needs OCR         : {len(ocr_records)}"
    )

    print(
        f"Empty records     : {len(empty_text_records)}"
    )

    print(
        f"Parsing errors    : {len(errors)}"
    )

    print()
    print(
        f"Output:\n{DOCUMENTS_FILE}"
    )

    print()
    print(
        f"Summary:\n{SUMMARY_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()