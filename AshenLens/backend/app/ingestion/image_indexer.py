import hashlib
import re
from pathlib import Path

from PIL import Image

from backend.app.schemas.document import CorpusFile
from backend.app.schemas.visual import VisualAsset


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def create_asset_id(source_path: str) -> str:
    """
    Create a deterministic visual asset ID.
    """

    digest = hashlib.sha1(
        source_path.encode("utf-8")
    ).hexdigest()[:16]

    return f"asset_{digest}"


def slug_to_title(slug: str) -> str:
    """
    Convert:

        the_weeping_lurker

    into:

        The Weeping Lurker
    """

    return (
        slug
        .replace("-", "_")
        .replace("_", " ")
        .strip()
        .title()
    )


# ---------------------------------------------------------
# Filename understanding
# ---------------------------------------------------------


def infer_asset_metadata(
    file_name: str,
) -> dict:
    """
    Infer entity and visual type from competition
    asset filenames.

    Examples:

        plate_08_creature_weeping_lurker.png

        atmo_heraldry_faction_house_morvain.png

        atmo_portrait_character_ignatz_ashgrove_the_oathless.png

        atmo_creature_creature_weeping_lurker.png
    """

    stem = Path(file_name).stem.lower()

    # -----------------------------------------------------
    # Official figure plates
    # -----------------------------------------------------

    plate_match = re.match(
        r"^plate_(\d+)_([a-z]+)_(.+)$",
        stem,
    )

    if plate_match:
        plate_number = int(
            plate_match.group(1)
        )

        subject_type = (
            plate_match.group(2)
        )

        entity_slug = (
            plate_match.group(3)
        )

        return {
            "asset_type": "plate",
            "entity_slug": entity_slug,
            "entity_name": slug_to_title(
                entity_slug
            ),
            "plate_number": plate_number,
            "subject_type": subject_type,
        }

    # -----------------------------------------------------
    # Wiki atmosphere / visual assets
    # -----------------------------------------------------
    #
    # IMPORTANT:
    # More specific prefixes MUST come BEFORE
    # generic prefixes.
    #
    # Example:
    #
    # atmo_creature_creature_weeping_lurker
    #
    # must match:
    #
    # atmo_creature_creature_
    #
    # BEFORE:
    #
    # atmo_creature_
    #
    # Otherwise the entity becomes:
    #
    # Creature Weeping Lurker
    #
    # instead of:
    #
    # Weeping Lurker
    # -----------------------------------------------------

    known_prefixes = [
        # -------------------------------------------------
        # Most specific prefixes first
        # -------------------------------------------------

        (
            "atmo_heraldry_faction_",
            "heraldry",
        ),

        (
            "atmo_portrait_character_",
            "portrait",
        ),

        (
            "atmo_relic_artifact_",
            "artifact",
        ),

        (
            "atmo_creature_creature_",
            "creature",
        ),

        (
            "atmo_landscape_location_",
            "landscape",
        ),

        (
            "atmo_battle_painting_conflict_",
            "battle_painting",
        ),

        # -------------------------------------------------
        # General / generic prefixes
        # -------------------------------------------------

        (
            "atmo_creature_",
            "creature",
        ),

        (
            "atmo_location_",
            "landscape",
        ),

        (
            "atmo_faction_",
            "heraldry",
        ),

        (
            "atmo_character_",
            "portrait",
        ),

        (
            "atmo_artifact_",
            "artifact",
        ),
    ]

    # -----------------------------------------------------
    # Match known prefixes
    # -----------------------------------------------------

    for prefix, asset_type in known_prefixes:

        if stem.startswith(prefix):

            entity_slug = (
                stem[len(prefix):]
                .strip("_")
            )

            return {
                "asset_type": asset_type,
                "entity_slug": entity_slug,
                "entity_name": slug_to_title(
                    entity_slug
                ),
            }

    # -----------------------------------------------------
    # Generic atmosphere image
    # -----------------------------------------------------

    if stem.startswith("atmo_"):

        entity_slug = (
            stem[len("atmo_"):]
            .strip("_")
        )

        return {
            "asset_type": "illustration",
            "entity_slug": entity_slug,
            "entity_name": slug_to_title(
                entity_slug
            ),
        }

    # -----------------------------------------------------
    # Generic fallback
    # -----------------------------------------------------

    return {
        "asset_type": "image",
        "entity_slug": stem,
        "entity_name": slug_to_title(
            stem
        ),
    }


# ---------------------------------------------------------
# Related wiki document
# ---------------------------------------------------------


def find_related_document(
    corpus_dir: Path,
    entity_slug: str | None,
) -> str | None:
    """
    Attempt to connect an image to its wiki article.

    Example:

        entity:
            weeping_lurker

        wiki:
            wiki/weeping_lurker.md
    """

    if not entity_slug:
        return None

    candidate = (
        corpus_dir
        / "wiki"
        / f"{entity_slug}.md"
    )

    if candidate.exists():

        return (
            candidate
            .relative_to(corpus_dir)
            .as_posix()
        )

    return None


# ---------------------------------------------------------
# Visual asset builder
# ---------------------------------------------------------


def build_visual_asset(
    corpus_file: CorpusFile,
    corpus_dir: Path,
) -> VisualAsset:
    """
    Build a normalized VisualAsset from
    one discovered image file.
    """

    path = Path(
        corpus_file.absolute_path
    )

    metadata = infer_asset_metadata(
        corpus_file.file_name
    )

    # -----------------------------------------------------
    # Read image information
    # -----------------------------------------------------

    with Image.open(path) as image:

        width, height = image.size

        image_format = image.format

        image_mode = image.mode

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    entity_slug = metadata.get(
        "entity_slug"
    )

    entity_name = metadata.get(
        "entity_name"
    )

    asset_type = metadata.get(
        "asset_type",
        "image",
    )

    # -----------------------------------------------------
    # Find related wiki document
    # -----------------------------------------------------

    related_document = (
        find_related_document(
            corpus_dir,
            entity_slug,
        )
    )

    # -----------------------------------------------------
    # Searchable natural-language representation
    # -----------------------------------------------------

    search_parts = [
        entity_name or "",
        asset_type,
        Path(
            corpus_file.file_name
        ).stem.replace("_", " "),
        corpus_file.category,
    ]

    if related_document:

        search_parts.append(
            related_document
            .replace("_", " ")
            .replace("/", " ")
        )

    search_text = " ".join(
        part
        for part in search_parts
        if part
    ).strip()

    # -----------------------------------------------------
    # Extra metadata
    # -----------------------------------------------------

    extra_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {
            "asset_type",
            "entity_slug",
            "entity_name",
        }
    }

    extra_metadata.update(
        {
            "image_format": image_format,
            "image_mode": image_mode,
        }
    )

    # -----------------------------------------------------
    # Build VisualAsset
    # -----------------------------------------------------

    return VisualAsset(
        asset_id=create_asset_id(
            corpus_file.source_path
        ),

        source_path=(
            corpus_file.source_path
        ),

        absolute_path=(
            corpus_file.absolute_path
        ),

        file_name=(
            corpus_file.file_name
        ),

        extension=(
            corpus_file.extension
        ),

        category=(
            corpus_file.category
        ),

        asset_type=asset_type,

        entity_slug=entity_slug,

        entity_name=entity_name,

        width=width,

        height=height,

        related_document=(
            related_document
        ),

        search_text=search_text,

        metadata=extra_metadata,
    )