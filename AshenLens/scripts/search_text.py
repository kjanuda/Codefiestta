import argparse
import sys

from pathlib import Path


ROOT_DIR = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT_DIR),
)


from backend.app.retrieval.semantic_retriever import (
    SemanticRetriever,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Search the Ashen Era "
            "text corpus semantically."
        )
    )

    parser.add_argument(
        "query",
        type=str,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    print("=" * 78)
    print(
        "AshenLens - Semantic "
        "Text Retrieval"
    )
    print("=" * 78)

    print()
    print(
        f"Query: {args.query}"
    )
    print()

    retriever = (
        SemanticRetriever()
    )

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
    )

    if not results:

        print(
            "No results found."
        )
        return

    for index, result in enumerate(
        results,
        start=1,
    ):

        chunk = result.chunk

        print(
            f"[{index}] "
            f"Score: "
            f"{result.score:.4f}"
        )

        print(
            f"    Dense   : "
            f"{result.dense_score:.4f}"
        )

        print(
            f"    Lexical : "
            f"{result.lexical_score:.4f}"
        )

        if (
            result.rerank_score
            is not None
        ):
            print(
                f"    Rerank  : "
                f"{result.rerank_score:.4f}"
            )

        print(
            f"    Source  : "
            f"{chunk['source_path']}"
        )

        if (
            chunk.get(
                "page_number"
            )
            is not None
        ):

            print(
                f"    Page    : "
                f"{chunk['page_number']}"
            )

        print(
            f"    Chunk   : "
            f"{chunk['chunk_index']}"
        )

        preview = (
            chunk["text"]
            .replace(
                "\n",
                " ",
            )
        )

        if len(preview) > 500:

            preview = (
                preview[:500]
                + "..."
            )

        print(
            f"    Text    : "
            f"{preview}"
        )

        print()


if __name__ == "__main__":
    main()