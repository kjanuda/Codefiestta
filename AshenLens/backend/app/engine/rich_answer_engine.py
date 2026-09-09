import base64
import mimetypes

from pathlib import Path

from backend.app.retrieval.semantic_retriever import (
    SemanticRetriever,
)

from backend.app.retrieval.visual_retriever import (
    VisualRetriever,
)

from backend.app.llm.openrouter_client import (
    OpenRouterClient,
)


ROOT_DIR = Path(
    __file__
).resolve().parents[3]


class RichAnswerEngine:

    VISUAL_CONFIDENCE_THRESHOLD = 60.0

    def __init__(self):

        self.text_retriever = (
            SemanticRetriever()
        )

        self.visual_retriever = (
            VisualRetriever(
                ROOT_DIR
                / "storage"
                / "visual_assets.jsonl"
            )
        )

        self.client = (
            OpenRouterClient()
        )

    def _asset_value(
        self,
        asset,
        key: str,
        default=None,
    ):

        if isinstance(
            asset,
            dict,
        ):
            return asset.get(
                key,
                default,
            )

        return getattr(
            asset,
            key,
            default,
        )

    def _image_to_data_url(
        self,
        image_path: Path,
    ) -> str:

        mime_type, _ = (
            mimetypes.guess_type(
                image_path.name
            )
        )

        mime_type = (
            mime_type
            or "image/png"
        )

        encoded = (
            base64.b64encode(
                image_path.read_bytes()
            ).decode("utf-8")
        )

        return (
            f"data:{mime_type};"
            f"base64,{encoded}"
        )

    def _build_text_context(
        self,
        results,
    ) -> str:

        blocks = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            chunk = result.chunk

            blocks.append(
                f"""
TEXT SOURCE {index}
Source: {chunk.get('source_path')}
Page: {chunk.get('page_number')}
Content:
{chunk.get('text')}
""".strip()
            )

        return "\n\n".join(
            blocks
        )

    def answer(
        self,
        question: str,
    ) -> dict:

        # --------------------------------
        # 1. Retrieve text evidence
        # --------------------------------

        text_results = (
            self.text_retriever.search(
                query=question,
                top_k=4,
                candidate_k=30,
                use_reranker=True,
            )
        )

        # --------------------------------
        # 2. Retrieve visual evidence
        # --------------------------------

        visual_results = (
            self.visual_retriever.search(
                question,
                top_k=3,
            )
        )

        best_visual = (
            visual_results[0]
            if visual_results
            else None
        )

        use_visual = bool(
            best_visual
            and best_visual.score
            >= self.VISUAL_CONFIDENCE_THRESHOLD
        )

        # --------------------------------
        # 3. Build text context
        # --------------------------------

        text_context = (
            self._build_text_context(
                text_results
            )
        )

        # --------------------------------
        # 4. System prompt
        # --------------------------------

        system_prompt = """
You are AshenLens, a grounded multimodal research assistant
for the fictional Ashen Era Archive.

You receive:
- a user question;
- retrieved archive text;
- optionally a retrieved archive image.

Evidence policy:

1. Use only the supplied archive evidence.
2. Never invent facts.
3. For questions asking about a figure plate, portrait, banner, illustration,
   diagram, table, visible symbol, numerical plate value, motif, or object:
   the visual evidence is PRIMARY.
4. Text evidence may omit information that is present only
   in the associated visual.
5. If text says a value is absent but the official visual
   clearly contains the requested value, report the visible
   value and explain that it comes from the visual plate.
6. Do not allow a semantically similar entity in retrieved
   text to replace the entity named in the question.
7. Keep the response concise and factual.
8. If a question uses wording that does not exactly match the visual label,
   return the exact visible value for the named entity without inventing a new unit.
9. If the visual shows only a number, return only the number.
10. Do not restate the user's unit wording unless that same unit is visible
    in the evidence.
11. Prefer the shortest correct grounded answer.

Return your response in this exact format:

ANSWER: <short direct answer only>
EXPLANATION: <1-2 sentence grounded explanation using only supplied evidence>
""".strip()

        # --------------------------------
        # 5. Build user content
        # --------------------------------

        user_text = f"""
Question:
{question}

Retrieved text evidence:
{text_context}
""".strip()

        content = [
            {
                "type": "text",
                "text": user_text,
            }
        ]

        # --------------------------------
        # 6. Visual metadata
        # --------------------------------

        visual_source = None
        visual_entity = None
        visual_score = None

        # --------------------------------
        # 7. Attach visual evidence
        # --------------------------------

        if use_visual:

            asset = best_visual.asset

            absolute_path = (
                self._asset_value(
                    asset,
                    "absolute_path",
                )
            )

            if not absolute_path:
                raise RuntimeError(
                    "Retrieved visual is missing "
                    "absolute_path."
                )

            image_path = Path(
                absolute_path
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Retrieved visual does not exist: "
                    f"{image_path}"
                )

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            self._image_to_data_url(
                                image_path
                            )
                        )
                    },
                }
            )

            visual_source = (
                self._asset_value(
                    asset,
                    "source_path",
                )
            )

            visual_entity = (
                self._asset_value(
                    asset,
                    "entity_name",
                )
            )

            visual_score = float(
                best_visual.score
            )

        # --------------------------------
        # 8. Build LLM messages
        # --------------------------------

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": content,
            },
        ]

        # --------------------------------
        # 9. Generate answer
        # --------------------------------

        response = self.client.chat(
            messages=messages,
            max_tokens=350,
            temperature=0.0,
        )

        # --------------------------------
        # 10. Collect text source metadata
        # --------------------------------

        text_sources = []

        for result in text_results:

            chunk = result.chunk

            text_sources.append(
                {
                    "source_path": (
                        chunk.get(
                            "source_path"
                        )
                    ),

                    "page_number": (
                        chunk.get(
                            "page_number"
                        )
                    ),

                    "score": round(
                        result.score,
                        4,
                    ),
                }
            )

        # --------------------------------
        # 11. Return result
        # --------------------------------

        return {
            "question": question,

            "response": response,

            "model": (
                self.client.last_model
            ),

            "visual": {
                "used": use_visual,

                "source_path": (
                    visual_source
                ),

                "entity": (
                    visual_entity
                ),

                "score": (
                    visual_score
                ),
            },

            "text_sources": (
                text_sources
            ),
        }