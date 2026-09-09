from typing import Any, Literal

from pydantic import BaseModel, Field


class CorpusFile(BaseModel):
    """
    Represents one physical file discovered inside the archive.
    """

    source_path: str
    absolute_path: str
    file_name: str
    extension: str
    category: str

    kind: Literal[
        "text",
        "image",
        "other",
    ]


class TextRecord(BaseModel):
    """
    A normalized text record produced by the ingestion pipeline.

    PDFs generate one record per page.
    DOCX, Markdown, and TXT files currently generate one record
    per document.
    """

    record_id: str
    document_id: str

    source_path: str
    file_name: str
    extension: str
    category: str

    page_number: int | None = None

    text: str
    char_count: int

    needs_ocr: bool = False
    image_count: int = 0

    metadata: dict[str, Any] = Field(default_factory=dict)