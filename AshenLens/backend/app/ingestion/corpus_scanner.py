from pathlib import Path

from backend.app.schemas.document import CorpusFile


TEXT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".md",
    ".txt",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def get_file_kind(extension: str) -> str:
    """
    Classify files into text documents, images, or other files.
    """

    extension = extension.lower()

    if extension in TEXT_EXTENSIONS:
        return "text"

    if extension in IMAGE_EXTENSIONS:
        return "image"

    return "other"


def get_category(relative_path: Path) -> str:
    """
    Determine the archive category from the first directory.

    Examples:
        chronicles/book.pdf -> chronicles
        wiki/article.md      -> wiki
        README.txt           -> root
    """

    if len(relative_path.parts) <= 1:
        return "root"

    return relative_path.parts[0]


def scan_corpus(corpus_dir: Path) -> list[CorpusFile]:
    """
    Recursively scan the Ashen Era Archive.

    Returns normalized metadata for every physical file.
    """

    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory does not exist: {corpus_dir}"
        )

    discovered_files: list[CorpusFile] = []

    files = sorted(
        (
            path
            for path in corpus_dir.rglob("*")
            if path.is_file()
        ),
        key=lambda path: str(path).lower(),
    )

    for file_path in files:
        relative_path = file_path.relative_to(corpus_dir)

        extension = file_path.suffix.lower()

        discovered_files.append(
            CorpusFile(
                source_path=relative_path.as_posix(),
                absolute_path=str(file_path.resolve()),
                file_name=file_path.name,
                extension=extension,
                category=get_category(relative_path),
                kind=get_file_kind(extension),
            )
        )

    return discovered_files