from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=2000,
    )


class TextSourceResponse(BaseModel):
    source_path: str | None = None
    page_number: int | None = None
    score: float | None = None


class VisualEvidenceResponse(BaseModel):
    used: bool
    source_path: str | None = None
    image_url: str | None = None
    entity: str | None = None
    score: float | None = None


class AskResponse(BaseModel):
    question: str

    answer: str
    explanation: str

    raw_response: str

    model: str | None = None

    visual: VisualEvidenceResponse

    text_sources: list[
        TextSourceResponse
    ]