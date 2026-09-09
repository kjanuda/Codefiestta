from typing import Any

from pydantic import BaseModel


class TextChunk(BaseModel):
    chunk_id: str
    record_id: str
    document_id: str

    source_path: str
    file_name: str
    extension: str
    category: str

    page_number: int | None = None

    chunk_index: int
    text: str
    char_count: int

    metadata: dict[str, Any] = {}