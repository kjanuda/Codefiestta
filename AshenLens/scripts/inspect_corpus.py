import json

from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

CORPUS_DIR = ROOT_DIR / "corpus" / "Ashen_Era_Archive"

QUESTIONS_FILE = CORPUS_DIR / "sample_questions.json"


def main():
    print("=" * 60)
    print("AshenLens - Corpus Inspection")
    print("=" * 60)

    if not CORPUS_DIR.exists():
        print(f"[ERROR] Corpus not found:")
        print(CORPUS_DIR)
        return

    files = [
        file
        for file in CORPUS_DIR.rglob("*")
        if file.is_file()
    ]

    extension_counts = Counter(
        file.suffix.lower() or "<no extension>"
        for file in files
    )

    print()
    print(f"Corpus path : {CORPUS_DIR}")
    print(f"Total files : {len(files)}")

    print()
    print("File types:")
    for extension, count in sorted(extension_counts.items()):
        print(f"  {extension:<10} {count}")

    if not QUESTIONS_FILE.exists():
        print()
        print("[ERROR] sample_questions.json not found.")
        return

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    track_counts = Counter(
        question.get("track", "UNKNOWN")
        for question in questions
    )

    print()
    print(f"Sample questions: {len(questions)}")

    print()
    print("Questions by track:")

    for track, count in track_counts.items():
        print(f"  {track}: {count}")

    print()
    print("1A questions:")
    print("-" * 60)

    for question in questions:
        if question.get("track", "").startswith("1A"):
            print(
                f"{question['qid']}: "
                f"{question['question']}"
            )

    print()
    print("=" * 60)
    print("Corpus inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()