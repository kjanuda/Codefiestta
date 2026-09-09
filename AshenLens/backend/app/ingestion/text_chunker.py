import hashlib
import re

from backend.app.schemas.document import TextRecord
from backend.app.schemas.chunk import TextChunk


MAX_CHARS = 1800
OVERLAP_CHARS = 250
MIN_CHARS = 80


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def make_chunk_id(
    record_id: str,
    chunk_index: int,
) -> str:

    raw = (
        f"{record_id}:"
        f"{chunk_index}"
    )

    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()


def split_text(
    text: str,
) -> list[str]:

    text = normalize_whitespace(
        text
    )

    if not text:
        return []

    if len(text) <= MAX_CHARS:
        return [text]

    chunks: list[str] = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + MAX_CHARS,
            text_length,
        )

        # Prefer a natural paragraph boundary.
        if end < text_length:

            paragraph_break = (
                text.rfind(
                    "\n\n",
                    start,
                    end,
                )
            )

            sentence_break = (
                text.rfind(
                    ". ",
                    start,
                    end,
                )
            )

            candidate = max(
                paragraph_break,
                sentence_break,
            )

            # Don't make a tiny chunk just
            # because an early boundary exists.
            if (
                candidate
                > start + (MAX_CHARS // 2)
            ):
                if candidate == sentence_break:
                    end = candidate + 1
                else:
                    end = candidate

        chunk = (
            text[start:end]
            .strip()
        )

        if (
            chunk
            and (
                len(chunk) >= MIN_CHARS
                or not chunks
            )
        ):
            chunks.append(
                chunk
            )

        if end >= text_length:
            break

        next_start = max(
            0,
            end - OVERLAP_CHARS,
        )

        # Avoid infinite loops.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def chunk_record(
    record: TextRecord,
) -> list[TextChunk]:

    # OCR-needed/empty records currently
    # contain no useful searchable text.
    if not record.text.strip():
        return []

    pieces = split_text(
        record.text
    )

    chunks: list[TextChunk] = []

    for index, piece in enumerate(
        pieces
    ):

        chunks.append(
            TextChunk(
                chunk_id=make_chunk_id(
                    record.record_id,
                    index,
                ),
                record_id=record.record_id,
                document_id=(
                    record.document_id
                ),
                source_path=(
                    record.source_path
                ),
                file_name=(
                    record.file_name
                ),
                extension=(
                    record.extension
                ),
                category=(
                    record.category
                ),
                page_number=(
                    record.page_number
                ),
                chunk_index=index,
                text=piece,
                char_count=len(piece),
                metadata={
                    "needs_ocr":
                        record.needs_ocr,
                    "image_count":
                        record.image_count,
                },
            )
        )

    return chunks