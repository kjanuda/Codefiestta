import base64
import os
import time

from pathlib import Path

from dotenv import load_dotenv
from google import genai


ROOT_DIR = Path(
    __file__
).resolve().parents[3]

load_dotenv(
    ROOT_DIR / ".env",
    override=True,
)


class GeminiError(RuntimeError):
    pass


class GeminiVisionClient:

    def __init__(self):

        self.api_key = os.getenv(
            "GEMINI_API_KEY",
            "",
        ).strip()

        self.primary_model = os.getenv(
            "GEMINI_VISION_MODEL",
            "gemini-3.8-flash",
        ).strip()

        self.fallback_model = os.getenv(
            "GEMINI_VISION_FALLBACK_MODEL",
            "gemini-3.7-flash",
        ).strip()

        if not self.api_key:
            raise GeminiError(
                "GEMINI_API_KEY is missing. "
                "Add it to the project .env file."
            )

        self.client = genai.Client(
            api_key=self.api_key
        )

        self.last_model: str | None = None

    def get_models(self) -> list[str]:

        return list(
            dict.fromkeys(
                [
                    self.primary_model,
                    self.fallback_model,
                ]
            )
        )

    def _call_model(
        self,
        model: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> str:

        image_base64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        retry_waits = [
            3,
            8,
        ]

        last_error = None

        for attempt in range(
            len(retry_waits) + 1
        ):

            try:

                interaction = (
                    self.client.interactions.create(
                        model=model,
                        input=[
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image",
                                "data": image_base64,
                                "mime_type": mime_type,
                            },
                        ],
                    )
                )

                answer = (
                    interaction.output_text
                    or ""
                ).strip()

                if answer:

                    self.last_model = model
                    return answer

                last_error = (
                    "Gemini returned an "
                    "empty response."
                )

            except Exception as error:

                last_error = str(error)

            if attempt < len(
                retry_waits
            ):

                wait_seconds = (
                    retry_waits[attempt]
                )

                print(
                    f"[{model}] request failed. "
                    f"Retrying in "
                    f"{wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

        raise GeminiError(
            f"{model} failed.\n"
            f"Last error: {last_error}"
        )

    def analyze_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> str:

        errors = []

        models = self.get_models()

        for index, model in enumerate(
            models,
            start=1,
        ):

            print(
                f"Gemini vision model "
                f"{index}/{len(models)}: "
                f"{model}"
            )

            try:

                return self._call_model(
                    model=model,
                    prompt=prompt,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                )

            except GeminiError as error:

                errors.append(
                    str(error)
                )

                if index < len(models):

                    print(
                        "Gemini model failed. "
                        "Trying fallback..."
                    )

        raise GeminiError(
            "All Gemini vision models failed.\n\n"
            + "\n\n".join(errors)
        )