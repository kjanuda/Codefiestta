import mimetypes

from pathlib import Path

from backend.app.llm.gemini_client import (
    GeminiVisionClient,
    GeminiError,
)


SYSTEM_PROMPT = """
You are the visual evidence reader for AshenLens.

AshenLens is a document research assistant for a completely
fictional archive created for a university AI competition.

Your job is to answer the user's question from the supplied visual.

Rules:

1. Use the supplied image as the factual evidence source.

2. Inspect all visible text, labels, numbers, symbols, objects,
   tables, diagrams, charts, artwork, and annotations carefully.

3. Match the user's wording to semantically equivalent labels
   visible in the image.

   The user's terminology may not be identical to the wording
   printed in the visual.

4. Do NOT return INSUFFICIENT_VISUAL_EVIDENCE merely because
   the question uses a different synonym, unit description,
   or phrasing from the image.

5. For numerical figure-plate questions:
   - identify the named entity in the question;
   - locate the row, bar, label, or annotation associated with
     that entity;
   - return the value attached to that entity;
   - do not confuse comparison/reference values with the
     requested entity's value.

6. For portraits:
   identify the object visibly held by the named person.

7. For heraldry:
   identify the visible emblem or symbol requested.

8. For artifact illustrations:
   identify visible motifs, engravings, or design features.

9. Return only a short factual answer.

10. Preserve the visible numerical value exactly.

11. If the requested information truly cannot be determined
    from the image after semantic matching, return exactly:
    INSUFFICIENT_VISUAL_EVIDENCE

Never invent a value that is not visible in the image.
""".strip()


INVALID_EXACT = {
    "safe",
    "unsafe",
    "user safety: safe",
    "user safety: unsafe",
}


def validate_answer(
    answer: str,
) -> str:

    answer = answer.strip()

    if not answer:

        raise GeminiError(
            "Vision model returned "
            "an empty answer."
        )

    normalized = answer.lower()

    if normalized in INVALID_EXACT:

        raise GeminiError(
            "Invalid safety-classifier "
            "response."
        )

    if (
        "user safety:" in normalized
        or "safety categories:"
        in normalized
    ):

        raise GeminiError(
            "Invalid safety-classifier "
            "response."
        )

    return answer


class VisionAnalyzer:

    def __init__(self):

        self.client = (
            GeminiVisionClient()
        )

    def answer_question(
        self,
        question: str,
        image_path: Path,
        entity_name: str | None = None,
        asset_type: str | None = None,
    ) -> str:

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        mime_type, _ = (
            mimetypes.guess_type(
                image_path.name
            )
        )

        if not mime_type:
            mime_type = "image/png"

        context_lines = []

        if entity_name:
            context_lines.append(
                f"Archive entity: "
                f"{entity_name}"
            )

        if asset_type:
            context_lines.append(
                f"Visual type: "
                f"{asset_type}"
            )

        context = "\n".join(
            context_lines
        )

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{context}\n\n"
            f"Question:\n"
            f"{question}\n\n"
            "Inspect the supplied archive "
            "image carefully and return "
            "only the answer."
        )

        answer = (
            self.client.analyze_image(
                prompt=prompt,
                image_bytes=(
                    image_path.read_bytes()
                ),
                mime_type=mime_type,
            )
        )

        return validate_answer(
            answer
        )