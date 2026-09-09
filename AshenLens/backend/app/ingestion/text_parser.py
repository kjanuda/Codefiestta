import hashlib
from pathlib import Path
from typing import Iterator

import pymupdf

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from backend.app.schemas.document import (
    CorpusFile,
    TextRecord,
)


def create_stable_id(prefix: str, value: str) -> str:
    """
    Create a deterministic ID.

    The same source path will always produce the same ID,
    making indexing reproducible between runs.
    """

    digest = hashlib.sha1(
        value.encode("utf-8")
    ).hexdigest()[:16]

    return f"{prefix}_{digest}"


def read_plain_text(path: Path) -> str:
    """
    Read Markdown/TXT files with safe encoding fallbacks.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
    ]

    for encoding in encodings:
        try:
            return path.read_text(
                encoding=encoding
            )
        except UnicodeDecodeError:
            continue

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def iter_docx_blocks(
    document: DocxDocument,
) -> Iterator[Paragraph | Table]:
    """
    Iterate through DOCX paragraphs and tables while
    preserving their document order.
    """

    body = document.element.body

    for child in body.iterchildren():

        if isinstance(child, CT_P):
            yield Paragraph(child, document)

        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def extract_docx_text(
    path: Path,
) -> tuple[str, int]:
    """
    Extract paragraphs and table contents from DOCX files.

    Returns:
        text
        embedded image count
    """

    document = Document(path)

    blocks: list[str] = []

    for block in iter_docx_blocks(document):

        if isinstance(block, Paragraph):
            text = block.text.strip()

            if text:
                blocks.append(text)

        elif isinstance(block, Table):

            table_lines: list[str] = []

            for row in block.rows:

                cells = [
                    cell.text.strip()
                    for cell in row.cells
                ]

                table_lines.append(
                    " | ".join(cells)
                )

            if table_lines:
                blocks.append(
                    "\n".join(table_lines)
                )

    image_count = sum(
        1
        for relationship in document.part.rels.values()
        if "image" in relationship.reltype
    )

    text = "\n\n".join(blocks).strip()

    return text, image_count


def parse_pdf(
    corpus_file: CorpusFile,
) -> list[TextRecord]:
    """
    Extract text from every PDF page.

    One PDF page becomes one TextRecord.
    """

    path = Path(corpus_file.absolute_path)

    document_id = create_stable_id(
        "doc",
        corpus_file.source_path,
    )

    records: list[TextRecord] = []

    with pymupdf.open(path) as pdf:

        page_count = len(pdf)

        for page_index in range(page_count):

            page = pdf[page_index]

            page_number = page_index + 1

            text = page.get_text(
                "text",
                sort=True,
            ).strip()

            image_count = len(
                page.get_images(full=True)
            )

            needs_ocr = (
                len(text) == 0
                and image_count > 0
            )

            record_id = create_stable_id(
                "record",
                (
                    f"{corpus_file.source_path}"
                    f"#page={page_number}"
                ),
            )

            records.append(
                TextRecord(
                    record_id=record_id,
                    document_id=document_id,

                    source_path=corpus_file.source_path,
                    file_name=corpus_file.file_name,
                    extension=corpus_file.extension,
                    category=corpus_file.category,

                    page_number=page_number,

                    text=text,
                    char_count=len(text),

                    needs_ocr=needs_ocr,
                    image_count=image_count,

                    metadata={
                        "page_count": page_count,
                    },
                )
            )

    return records


def parse_docx(
    corpus_file: CorpusFile,
) -> list[TextRecord]:
    """
    Extract text and tables from a DOCX document.
    """

    path = Path(corpus_file.absolute_path)

    document_id = create_stable_id(
        "doc",
        corpus_file.source_path,
    )

    text, image_count = extract_docx_text(path)

    needs_ocr = (
        len(text) == 0
        and image_count > 0
    )

    record = TextRecord(
        record_id=create_stable_id(
            "record",
            corpus_file.source_path,
        ),
        document_id=document_id,

        source_path=corpus_file.source_path,
        file_name=corpus_file.file_name,
        extension=corpus_file.extension,
        category=corpus_file.category,

        page_number=None,

        text=text,
        char_count=len(text),

        needs_ocr=needs_ocr,
        image_count=image_count,

        metadata={},
    )

    return [record]


def parse_plain_document(
    corpus_file: CorpusFile,
) -> list[TextRecord]:
    """
    Parse Markdown and TXT documents.
    """

    path = Path(corpus_file.absolute_path)

    text = read_plain_text(path).strip()

    document_id = create_stable_id(
        "doc",
        corpus_file.source_path,
    )

    record = TextRecord(
        record_id=create_stable_id(
            "record",
            corpus_file.source_path,
        ),
        document_id=document_id,

        source_path=corpus_file.source_path,
        file_name=corpus_file.file_name,
        extension=corpus_file.extension,
        category=corpus_file.category,

        page_number=None,

        text=text,
        char_count=len(text),

        needs_ocr=False,
        image_count=0,

        metadata={},
    )

    return [record]


def parse_text_file(
    corpus_file: CorpusFile,
) -> list[TextRecord]:
    """
    Route a discovered file to the correct parser.
    """

    extension = corpus_file.extension.lower()

    if extension == ".pdf":
        return parse_pdf(corpus_file)

    if extension == ".docx":
        return parse_docx(corpus_file)

    if extension in {
        ".md",
        ".txt",
    }:
        return parse_plain_document(
            corpus_file
        )

    raise ValueError(
        f"Unsupported text extension: {extension}"
    )