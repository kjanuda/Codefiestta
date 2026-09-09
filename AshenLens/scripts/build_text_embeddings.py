import argparse
import json
import os
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import voyageai
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT_DIR),
)

load_dotenv(
    ROOT_DIR / ".env",
    override=True,
)


STORAGE_DIR = ROOT_DIR / "storage"

CHUNKS_PATH = (
    STORAGE_DIR
    / "text_chunks.jsonl"
)

EMBEDDINGS_PATH = (
    STORAGE_DIR
    / "text_embeddings.npy"
)

STATE_PATH = (
    STORAGE_DIR
    / "text_embeddings_state.json"
)

MANIFEST_PATH = (
    STORAGE_DIR
    / "text_embeddings_manifest.json"
)


DOCUMENT_MODEL = os.getenv(
    "VOYAGE_DOCUMENT_MODEL",
    "voyage-4-large",
).strip()

QUERY_MODEL = os.getenv(
    "VOYAGE_QUERY_MODEL",
    "voyage-4-lite",
).strip()

EMBEDDING_DIMENSION = int(
    os.getenv(
        "VOYAGE_EMBEDDING_DIMENSION",
        "1024",
    )
)


def load_chunks() -> list[dict]:

    chunks = []

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            chunks.append(
                json.loads(line)
            )

    return chunks


def save_json_atomic(
    path: Path,
    data: dict,
):

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp_path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )

    temp_path.replace(path)


def normalize_vectors(
    vectors: np.ndarray,
) -> np.ndarray:

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    return (
        vectors / norms
    )


def embed_batch(
    client: voyageai.Client,
    texts: list[str],
    retries: int = 5,
):

    waits = [
        2,
        4,
        8,
        16,
    ]

    last_error = None

    for attempt in range(
        retries
    ):

        try:

            return client.embed(
                texts,
                model=DOCUMENT_MODEL,
                input_type="document",
                output_dimension=(
                    EMBEDDING_DIMENSION
                ),
            )

        except Exception as error:

            last_error = error

            if attempt >= retries - 1:
                break

            wait_seconds = waits[
                min(
                    attempt,
                    len(waits) - 1,
                )
            ]

            print(
                f"Voyage request failed: "
                f"{error}"
            )

            print(
                f"Retrying in "
                f"{wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "Voyage embedding request "
        f"failed after retries: "
        f"{last_error}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
    )

    args = parser.parse_args()

    if not CHUNKS_PATH.exists():

        raise FileNotFoundError(
            f"Missing: {CHUNKS_PATH}"
        )

    api_key = os.getenv(
        "VOYAGE_API_KEY",
        "",
    ).strip()

    if not api_key:

        raise RuntimeError(
            "VOYAGE_API_KEY is missing "
            "from .env"
        )

    chunks = load_chunks()

    total_chunks = len(chunks)

    if total_chunks == 0:

        raise RuntimeError(
            "No text chunks found."
        )

    if args.fresh:

        for path in [
            EMBEDDINGS_PATH,
            STATE_PATH,
            MANIFEST_PATH,
        ]:

            if path.exists():
                path.unlink()

    client = voyageai.Client(
        api_key=api_key
    )

    start_index = 0
    total_tokens = 0

    if (
        STATE_PATH.exists()
        and EMBEDDINGS_PATH.exists()
    ):

        state = json.loads(
            STATE_PATH.read_text(
                encoding="utf-8"
            )
        )

        if (
            state.get("total_chunks")
            != total_chunks
        ):

            raise RuntimeError(
                "Chunk count changed. "
                "Run again with --fresh."
            )

        if (
            state.get("model")
            != DOCUMENT_MODEL
        ):

            raise RuntimeError(
                "Embedding model changed. "
                "Run again with --fresh."
            )

        if (
            state.get("dimension")
            != EMBEDDING_DIMENSION
        ):

            raise RuntimeError(
                "Embedding dimension changed. "
                "Run again with --fresh."
            )

        start_index = int(
            state.get(
                "completed",
                0,
            )
        )

        total_tokens = int(
            state.get(
                "total_tokens",
                0,
            )
        )

        embeddings = (
            np.lib.format.open_memmap(
                EMBEDDINGS_PATH,
                mode="r+",
            )
        )

        print(
            f"Resuming from chunk "
            f"{start_index:,}"
        )

    else:

        embeddings = (
            np.lib.format.open_memmap(
                EMBEDDINGS_PATH,
                mode="w+",
                dtype=np.float32,
                shape=(
                    total_chunks,
                    EMBEDDING_DIMENSION,
                ),
            )
        )

        state = {
            "model":
                DOCUMENT_MODEL,

            "query_model":
                QUERY_MODEL,

            "dimension":
                EMBEDDING_DIMENSION,

            "total_chunks":
                total_chunks,

            "completed":
                0,

            "total_tokens":
                0,
        }

        save_json_atomic(
            STATE_PATH,
            state,
        )

    print("=" * 72)
    print(
        "AshenLens - Voyage Text "
        "Embedding Builder"
    )
    print("=" * 72)

    print(
        f"Chunks          : "
        f"{total_chunks:,}"
    )

    print(
        f"Document model  : "
        f"{DOCUMENT_MODEL}"
    )

    print(
        f"Query model     : "
        f"{QUERY_MODEL}"
    )

    print(
        f"Dimensions      : "
        f"{EMBEDDING_DIMENSION}"
    )

    print(
        f"Batch size      : "
        f"{args.batch_size}"
    )

    print(
        f"Starting at     : "
        f"{start_index:,}"
    )

    print()

    for batch_start in range(
        start_index,
        total_chunks,
        args.batch_size,
    ):

        batch_end = min(
            batch_start
            + args.batch_size,
            total_chunks,
        )

        texts = [
            chunk["text"]
            for chunk
            in chunks[
                batch_start:
                batch_end
            ]
        ]

        result = embed_batch(
            client=client,
            texts=texts,
        )

        batch_vectors = np.asarray(
            result.embeddings,
            dtype=np.float32,
        )

        expected_shape = (
            len(texts),
            EMBEDDING_DIMENSION,
        )

        if (
            batch_vectors.shape
            != expected_shape
        ):

            raise RuntimeError(
                "Unexpected embedding "
                f"shape: "
                f"{batch_vectors.shape}. "
                f"Expected "
                f"{expected_shape}."
            )

        batch_vectors = (
            normalize_vectors(
                batch_vectors
            )
        )

        embeddings[
            batch_start:
            batch_end
        ] = batch_vectors

        embeddings.flush()

        batch_tokens = int(
            getattr(
                result,
                "total_tokens",
                0,
            )
            or 0
        )

        total_tokens += (
            batch_tokens
        )

        state = {
            "model":
                DOCUMENT_MODEL,

            "query_model":
                QUERY_MODEL,

            "dimension":
                EMBEDDING_DIMENSION,

            "total_chunks":
                total_chunks,

            "completed":
                batch_end,

            "total_tokens":
                total_tokens,

            "updated_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        save_json_atomic(
            STATE_PATH,
            state,
        )

        percent = (
            batch_end
            / total_chunks
            * 100
        )

        print(
            f"[{batch_end:4d}/"
            f"{total_chunks}] "
            f"{percent:6.2f}% | "
            f"batch tokens: "
            f"{batch_tokens:,} | "
            f"total tokens: "
            f"{total_tokens:,}"
        )

    manifest = {
        "index_type":
            "voyage_dense_embeddings",

        "document_model":
            DOCUMENT_MODEL,

        "query_model":
            QUERY_MODEL,

        "dimension":
            EMBEDDING_DIMENSION,

        "vector_count":
            total_chunks,

        "normalized":
            True,

        "dtype":
            "float32",

        "chunks_file":
            "storage/text_chunks.jsonl",

        "embeddings_file":
            "storage/text_embeddings.npy",

        "total_embedding_tokens":
            total_tokens,

        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    save_json_atomic(
        MANIFEST_PATH,
        manifest,
    )

    print()
    print("=" * 72)
    print(
        "Embedding index completed."
    )
    print("=" * 72)

    print(
        f"Vectors : "
        f"{EMBEDDINGS_PATH}"
    )

    print(
        f"Manifest: "
        f"{MANIFEST_PATH}"
    )

    print(
        f"Tokens  : "
        f"{total_tokens:,}"
    )

    print(
        f"Shape   : "
        f"({total_chunks}, "
        f"{EMBEDDING_DIMENSION})"
    )


if __name__ == "__main__":
    main()