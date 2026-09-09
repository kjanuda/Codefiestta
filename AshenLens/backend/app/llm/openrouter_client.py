import os
import time

from pathlib import Path

import httpx
from dotenv import load_dotenv


ROOT_DIR = Path(
    __file__
).resolve().parents[3]


load_dotenv(
    ROOT_DIR / ".env",
    override=True,
)


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:

    def __init__(self):

        self.api_key = os.getenv(
            "OPENROUTER_API_KEY",
            "",
        ).strip()

        if not self.api_key:

            raise OpenRouterError(
                "OPENROUTER_API_KEY is missing."
            )

        self.base_url = os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ).rstrip("/")

        self.timeout = float(
            os.getenv(
                "OPENROUTER_TIMEOUT",
                "120",
            )
        )

        self.primary_model = os.getenv(
            "VISION_MODEL",
            "google/gemini-2.5-flash",
        ).strip()

        fallback_raw = os.getenv(
            "VISION_FALLBACK_MODELS",
            "",
        )

        self.fallback_models = [
            model.strip()
            for model in fallback_raw.split(",")
            if model.strip()
        ]

        self.last_model: str | None = None

        # HTTP client is kept for compatibility
        # with the rest of the project.
        self.http = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization":
                    f"Bearer {self.api_key}",

                "Content-Type":
                    "application/json",

                "HTTP-Referer":
                    "http://localhost",

                "X-Title":
                    "AshenLens",
            },
        )

    def get_models(
        self,
    ) -> list[str]:

        models = [
            self.primary_model,
            *self.fallback_models,
        ]

        # Remove duplicates while
        # preserving order.
        return list(
            dict.fromkeys(
                model
                for model in models
                if model
            )
        )

    def _extract_content(
        self,
        data: dict,
    ) -> str:

        choices = data.get(
            "choices",
            [],
        )

        if not choices:
            return ""

        message = (
            choices[0]
            .get(
                "message",
                {},
            )
        )

        content = message.get(
            "content",
            "",
        )

        if isinstance(
            content,
            str,
        ):

            return content.strip()

        # Some models return
        # multimodal content blocks.
        if isinstance(
            content,
            list,
        ):

            text_parts = []

            for block in content:

                if not isinstance(
                    block,
                    dict,
                ):
                    continue

                text = block.get(
                    "text"
                )

                if text:

                    text_parts.append(
                        str(text)
                    )

            return "\n".join(
                text_parts
            ).strip()

        return ""

    def _is_invalid_output(
        self,
        content: str,
    ) -> bool:

        normalized = (
            content.lower()
            .strip()
        )

        invalid_exact = {
            "safe",
            "unsafe",
            "user safety: safe",
            "user safety: unsafe",
        }

        if normalized in invalid_exact:
            return True

        invalid_phrases = [
            "user safety:",
            "safety categories:",
        ]

        return any(
            phrase in normalized
            for phrase in invalid_phrases
        )

    def _call_model(
        self,
        model: str,
        messages: list,
        max_tokens: int,
        temperature: float,
    ) -> str:

        payload = {
            "model":
                model,

            "messages":
                messages,

            "temperature":
                temperature,

            "max_tokens":
                max_tokens,
        }

        retry_waits = [
            2,
            5,
            10,
        ]

        attempts = (
            len(retry_waits)
            + 1
        )

        last_error = None

        for attempt in range(
            attempts
        ):

            try:

                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization":
                            f"Bearer {self.api_key}",

                        "Content-Type":
                            "application/json",

                        "HTTP-Referer":
                            "http://localhost",

                        "X-Title":
                            "AshenLens",
                    },
                    json=payload,
                    timeout=self.timeout,
                )

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.ConnectError,
            ) as error:

                last_error = (
                    f"Network error: "
                    f"{error}"
                )

                if (
                    attempt
                    < len(retry_waits)
                ):

                    wait = (
                        retry_waits[
                            attempt
                        ]
                    )

                    print(
                        f"[{model}] "
                        f"network error. "
                        f"Retrying in "
                        f"{wait}s..."
                    )

                    time.sleep(
                        wait
                    )

                    continue

                break

            # -----------------------------
            # Authentication errors
            # Do NOT waste time retrying.
            # -----------------------------

            if response.status_code in {
                401,
                403,
            }:

                try:

                    body = (
                        response.json()
                    )

                except Exception:

                    body = (
                        response.text
                    )

                raise OpenRouterError(
                    f"HTTP "
                    f"{response.status_code}: "
                    f"{body}"
                )

            # -----------------------------
            # Temporary rate/server errors
            # -----------------------------

            if (
                response.status_code
                == 429
                or
                500
                <= response.status_code
                < 600
            ):

                last_error = (
                    f"HTTP "
                    f"{response.status_code}: "
                    f"{response.text}"
                )

                if (
                    attempt
                    < len(retry_waits)
                ):

                    wait = (
                        retry_waits[
                            attempt
                        ]
                    )

                    print(
                        f"[{model}] "
                        f"temporary HTTP "
                        f"{response.status_code}. "
                        f"Retrying in "
                        f"{wait}s..."
                    )

                    time.sleep(
                        wait
                    )

                    continue

                break

            # -----------------------------
            # Other errors such as
            # bad model id / bad request.
            # -----------------------------

            if (
                response.status_code
                >= 400
            ):

                raise OpenRouterError(
                    f"HTTP "
                    f"{response.status_code}: "
                    f"{response.text}"
                )

            try:

                data = response.json()

            except Exception as error:

                raise OpenRouterError(
                    "OpenRouter returned "
                    "invalid JSON: "
                    f"{error}"
                )

            content = (
                self._extract_content(
                    data
                )
            )

            if not content:

                raise OpenRouterError(
                    "Model returned "
                    "empty text content."
                )

            if self._is_invalid_output(
                content
            ):

                raise OpenRouterError(
                    "Model returned a "
                    "safety-classifier "
                    "response instead of "
                    "the requested answer."
                )

            actual_model = data.get(
                "model"
            )

            self.last_model = (
                actual_model
                or model
            )

            return content

        raise OpenRouterError(
            last_error
            or (
                f"Model '{model}' "
                "failed."
            )
        )

    def chat(
        self,
        messages: list,
        max_tokens: int = 350,
        temperature: float = 0.0,
    ) -> str:

        models = self.get_models()

        errors = []

        for index, model in enumerate(
            models,
            start=1,
        ):

            print(
                f"Vision provider "
                f"{index}/"
                f"{len(models)}: "
                f"{model}"
            )

            try:

                return self._call_model(
                    model=model,
                    messages=messages,
                    max_tokens=(
                        max_tokens
                    ),
                    temperature=(
                        temperature
                    ),
                )

            except OpenRouterError as error:

                errors.append(
                    f"{model} failed.\n"
                    f"Last error: {error}"
                )

                if (
                    index
                    < len(models)
                ):

                    print(
                        "Provider failed. "
                        "Trying fallback..."
                    )

        raise OpenRouterError(
            "All configured OpenRouter "
            "models failed.\n\n"
            + "\n\n".join(
                errors
            )
        )

    def close(self):

        self.http.close()

    def __del__(self):

        try:
            self.http.close()

        except Exception:
            pass