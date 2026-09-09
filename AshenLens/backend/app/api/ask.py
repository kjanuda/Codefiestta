import re

from functools import lru_cache

from fastapi import (
    APIRouter,
    HTTPException,
)

from backend.app.engine.rich_answer_engine import (
    RichAnswerEngine,
)

from backend.app.schemas.qa import (
    AskRequest,
    AskResponse,
    TextSourceResponse,
    VisualEvidenceResponse,
)


router = APIRouter()


@lru_cache(maxsize=1)
def get_engine() -> RichAnswerEngine:
    """
    Create the heavy retrieval/LLM engine once.

    This avoids reloading the 3920 text chunks
    and embedding matrix on every API request.
    """

    return RichAnswerEngine()


def parse_model_response(
    response: str,
) -> tuple[str, str]:

    response = response.strip()

    answer = ""
    explanation = ""

    answer_match = re.search(
        r"ANSWER:\s*(.+?)(?=\nEXPLANATION:|\Z)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    explanation_match = re.search(
        r"EXPLANATION:\s*(.+)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if answer_match:
        answer = (
            answer_match
            .group(1)
            .strip()
        )

    if explanation_match:
        explanation = (
            explanation_match
            .group(1)
            .strip()
        )

    # Fallback in case a provider ignores
    # our requested response format.
    if not answer:
        first_line = (
            response.splitlines()[0]
            if response
            else ""
        )

        answer = first_line.strip()

    return (
        answer,
        explanation,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
)
def ask_question(
    payload: AskRequest,
):

    question = (
        payload.question.strip()
    )

    try:

        engine = get_engine()

        result = engine.answer(
            question
        )

        raw_response = (
            result.get(
                "response",
                "",
            )
        )

        answer, explanation = (
            parse_model_response(
                raw_response
            )
        )

        visual = (
            result.get(
                "visual",
                {},
            )
        )

        visual_source = (
            visual.get(
                "source_path"
            )
        )

        image_url = None

        if (
            visual.get("used")
            and visual_source
        ):

            normalized_path = (
                visual_source
                .replace("\\", "/")
                .lstrip("/")
            )

            image_url = (
                f"/archive/"
                f"{normalized_path}"
            )

        text_sources = [
            TextSourceResponse(
                source_path=(
                    source.get(
                        "source_path"
                    )
                ),
                page_number=(
                    source.get(
                        "page_number"
                    )
                ),
                score=(
                    source.get(
                        "score"
                    )
                ),
            )
            for source
            in result.get(
                "text_sources",
                [],
            )
        ]

        return AskResponse(
            question=question,

            answer=answer,

            explanation=(
                explanation
            ),

            raw_response=(
                raw_response
            ),

            model=result.get(
                "model"
            ),

            visual=(
                VisualEvidenceResponse(
                    used=bool(
                        visual.get(
                            "used",
                            False,
                        )
                    ),

                    source_path=(
                        visual_source
                    ),

                    image_url=(
                        image_url
                    ),

                    entity=(
                        visual.get(
                            "entity"
                        )
                    ),

                    score=(
                        visual.get(
                            "score"
                        )
                    ),
                )
            ),

            text_sources=(
                text_sources
            ),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"AshenLens failed to "
                f"answer the question: "
                f"{error}"
            ),
        )