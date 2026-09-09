import json
import re

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


# ---------------------------------------------------------
# Data structure returned by the retriever
# ---------------------------------------------------------


@dataclass
class VisualSearchResult:
    asset: dict
    score: float
    reasons: list[str]


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "its",
    "known",
    "of",
    "on",
    "the",
    "their",
    "this",
    "to",
    "what",
    "which",
    "who",
    "with",
    "according",
    "depicting",
    "depicted",
    "illustrating",
    "illustration",
    "official",
    "figure",
    "plate",
}


def normalize_text(text: str) -> str:
    """
    Normalize text for lexical comparison.
    """

    text = text.lower()

    text = text.replace(
        "’",
        "'",
    )

    text = re.sub(
        r"[^a-z0-9\s'-]",
        " ",
        text,
    )

    text = text.replace(
        "-",
        " ",
    )

    text = text.replace(
        "'",
        "",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text: str) -> list[str]:
    """
    Convert text to useful search tokens.
    """

    normalized = normalize_text(text)

    return [
        token
        for token in normalized.split()
        if token not in STOP_WORDS
        and len(token) > 1
    ]


# ---------------------------------------------------------
# Query intent
# ---------------------------------------------------------


def infer_visual_type(
    question: str,
) -> str | None:
    """
    Infer what type of visual the question probably needs.
    """

    query = normalize_text(question)

    # Heraldry / banners / emblems
    if any(
        phrase in query
        for phrase in [
            "banner",
            "emblem",
            "heraldry",
            "crest",
            "symbol",
        ]
    ):
        return "heraldry"

    # Character portraits
    if any(
        phrase in query
        for phrase in [
            "portrait",
            "holding",
            "held",
            "wearing",
        ]
    ):
        return "portrait"

    # Explicit structured figure plates
    if any(
        phrase in query
        for phrase in [
            "figure plate",
            "threat classification",
            "numerical rating",
            "attunement cost",
            "shards of will",
            "garrison strength",
            "recorded total",
        ]
    ):
        return "plate"

    # Relic / artifact illustration
    if any(
        phrase in query
        for phrase in [
            "engraved",
            "motif",
            "artifact",
            "relic",
        ]
    ):
        return "artifact"

    return None


# ---------------------------------------------------------
# Retriever
# ---------------------------------------------------------


class VisualRetriever:

    def __init__(
        self,
        visual_index_path: Path,
    ):
        self.visual_index_path = (
            visual_index_path
        )

        self.assets = (
            self._load_assets()
        )

    def _load_assets(
        self,
    ) -> list[dict]:

        if not self.visual_index_path.exists():
            raise FileNotFoundError(
                "Visual index does not exist: "
                f"{self.visual_index_path}"
            )

        assets = []

        with self.visual_index_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                assets.append(
                    json.loads(line)
                )

        return assets

    def _score_asset(
        self,
        question: str,
        asset: dict,
    ) -> VisualSearchResult:

        query_normalized = (
            normalize_text(question)
        )

        query_tokens = set(
            tokenize(question)
        )

        entity_name = (
            asset.get("entity_name")
            or ""
        )

        entity_normalized = (
            normalize_text(entity_name)
        )

        search_text = (
            asset.get("search_text")
            or ""
        )

        search_tokens = set(
            tokenize(search_text)
        )

        expected_type = (
            infer_visual_type(question)
        )

        asset_type = (
            asset.get("asset_type")
            or ""
        )

        score = 0.0
        reasons = []

        # -------------------------------------------------
        # 1. Exact entity phrase
        # -------------------------------------------------

        if (
            entity_normalized
            and entity_normalized
            in query_normalized
        ):
            score += 100.0

            reasons.append(
                "exact entity phrase"
            )

        # -------------------------------------------------
        # 2. Entity token coverage
        # -------------------------------------------------

        entity_tokens = set(
            tokenize(entity_name)
        )

        if entity_tokens:

            matched_entity_tokens = (
                query_tokens
                & entity_tokens
            )

            coverage = (
                len(matched_entity_tokens)
                / len(entity_tokens)
            )

            score += coverage * 55.0

            if matched_entity_tokens:
                reasons.append(
                    "entity token match: "
                    + ", ".join(
                        sorted(
                            matched_entity_tokens
                        )
                    )
                )

        # -------------------------------------------------
        # 3. General token overlap
        # -------------------------------------------------

        overlap = (
            query_tokens
            & search_tokens
        )

        score += (
            len(overlap) * 4.0
        )

        if overlap:
            reasons.append(
                "search token overlap"
            )

        # -------------------------------------------------
        # 4. Fuzzy entity similarity
        # -------------------------------------------------

        if entity_normalized:

            fuzzy_score = (
                SequenceMatcher(
                    None,
                    query_normalized,
                    entity_normalized,
                ).ratio()
            )

            score += (
                fuzzy_score * 15.0
            )

        # -------------------------------------------------
        # 5. Visual-type intent
        # -------------------------------------------------

        if expected_type:

            if asset_type == expected_type:

                score += 35.0

                reasons.append(
                    f"visual type match: "
                    f"{expected_type}"
                )

            # Plate questions should strongly prefer
            # official figure plates.
            elif expected_type == "plate":

                score -= 10.0

        # -------------------------------------------------
        # 6. Prefer canonical codex plate
        #    over the duplicate top-level copy.
        # -------------------------------------------------

        if (
            asset_type == "plate"
            and asset.get("category")
            == "codex"
        ):
            score += 5.0

            reasons.append(
                "canonical codex plate"
            )

        return VisualSearchResult(
            asset=asset,
            score=score,
            reasons=reasons,
        )

    def search(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[VisualSearchResult]:
        """
        Rank visual assets for a natural-language question.
        """

        results = [
            self._score_asset(
                question,
                asset,
            )
            for asset in self.assets
        ]

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        # -------------------------------------------------
        # Deduplicate exact visual duplicates.
        #
        # We have figure plates duplicated under:
        # codex/images/
        # images/
        # -------------------------------------------------

        deduplicated = []

        seen_keys = set()

        for result in results:

            asset = result.asset

            key = (
                asset.get("file_name"),
                asset.get("entity_slug"),
                asset.get("asset_type"),
            )

            if key in seen_keys:
                continue

            seen_keys.add(key)

            deduplicated.append(
                result
            )

            if (
                len(deduplicated)
                >= top_k
            ):
                break

        return deduplicated