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


from backend.app.schemas.document import (
    TextRecord,
)

from backend.app.ingestion.text_chunker import (
    chunk_record,
)


STORAGE_DIR = (
    ROOT_DIR / "storage"
)

DOCUMENTS_PATH = (
    STORAGE_DIR
    / "documents.jsonl"
)

CHUNKS_PATH = (
    STORAGE_DIR
    / "text_chunks.jsonl"
)

SUMMARY_PATH = (
    STORAGE_DIR
    / "text_chunks_summary.json"
)


def load_records():
    with DOCUMENTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            yield TextRecord.model_validate(
                json.loads(line)
            )


def main():

    if not DOCUMENTS_PATH.exists():

        raise FileNotFoundError(
            f"Missing: {DOCUMENTS_PATH}"
        )

    STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    record_count = 0
    chunk_count = 0
    total_chars = 0

    categories: dict[str, int] = {}

    with CHUNKS_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:

        for record in load_records():

            record_count += 1

            chunks = chunk_record(
                record
            )

            for chunk in chunks:

                output.write(
                    chunk.model_dump_json()
                    + "\n"
                )

                chunk_count += 1

                total_chars += (
                    chunk.char_count
                )

                categories[
                    chunk.category
                ] = (
                    categories.get(
                        chunk.category,
                        0,
                    )
                    + 1
                )

    summary = {
        "input_records":
            record_count,

        "generated_chunks":
            chunk_count,

        "total_chunk_characters":
            total_chars,

        "average_chunk_characters":
            round(
                total_chars
                / max(chunk_count, 1),
                2,
            ),

        "categories":
            dict(
                sorted(
                    categories.items()
                )
            ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "AshenLens - Text Chunk Builder"
    )
    print("=" * 72)

    print(
        f"Input records      : "
        f"{record_count}"
    )

    print(
        f"Generated chunks   : "
        f"{chunk_count}"
    )

    print(
        f"Total characters   : "
        f"{total_chars:,}"
    )

    print(
        f"Average chunk chars : "
        f"{summary['average_chunk_characters']}"
    )

    print()
    print(
        f"Chunks  : {CHUNKS_PATH}"
    )
    print(
        f"Summary : {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()