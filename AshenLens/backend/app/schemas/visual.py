from typing import Any

from pydantic import BaseModel, Field


class VisualAsset(BaseModel):
    """
    Represents one standalone visual asset discovered
    inside the Ashen Era Archive.
    """

    asset_id: str

    source_path: str
    absolute_path: str
    file_name: str
    extension: str
    category: str

    asset_type: str

    entity_slug: str | None = None
    entity_name: str | None = None

    width: int
    height: int

    related_document: str | None = None

    search_text: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )