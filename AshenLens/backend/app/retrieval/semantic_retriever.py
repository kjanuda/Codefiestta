import json
import os
import re

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import voyageai
from dotenv import load_dotenv


ROOT_DIR = Path(
    __file__
).resolve().parents[3]

load_dotenv(
    ROOT_DIR / ".env",
    override=True,
)


STORAGE_DIR = ROOT_DIR / "storage"

CHUNKS_PATH = (
    STORAGE_DIR
    / "text_chunks.jsonl"
)

EMBEDDINGS_PATH = (
    STORAGE_DIR
    / "text_embeddings.npy"
)

MANIFEST_PATH = (
    STORAGE_DIR
    / "text_embeddings_manifest.json"
)


STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "what",
    "which",
    "who",
    "where",
    "when",
    "how",
    "is",
    "are",
    "was",
    "were",
    "according",
    "does",
    "did",
    "its",
    "it",
    "this",
    "that",
}


@dataclass
class SemanticSearchResult:
    chunk: dict
    dense_score: float
    lexical_score: float
    rerank_score: float | None
    score: float


def tokenize(
    text: str,
) -> set[str]:

    tokens = re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )

    return {
        token
        for token in tokens
        if (
            len(token) > 1
            and token
            not in STOP_WORDS
        )
    }


class SemanticRetriever:

    def __init__(self):

        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Missing: {CHUNKS_PATH}"
            )

        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"Missing: {EMBEDDINGS_PATH}"
            )

        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(
                f"Missing: {MANIFEST_PATH}"
            )

        api_key = os.getenv(
            "VOYAGE_API_KEY",
            "",
        ).strip()

        if not api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY is missing."
            )

        self.manifest = json.loads(
            MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )

        self.query_model = (
            self.manifest.get(
                "query_model",
                "voyage-4-lite",
            )
        )

        self.rerank_model = os.getenv(
            "VOYAGE_RERANK_MODEL",
            "rerank-2.5",
        ).strip()

        self.dimension = int(
            self.manifest.get(
                "dimension",
                1024,
            )
        )

        self.chunks = (
            self._load_chunks()
        )

        self.embeddings = np.load(
            EMBEDDINGS_PATH,
            mmap_mode="r",
        )

        expected_shape = (
            len(self.chunks),
            self.dimension,
        )

        if (
            self.embeddings.shape
            != expected_shape
        ):
            raise RuntimeError(
                "Chunk/vector mismatch. "
                f"Got {self.embeddings.shape}, "
                f"expected {expected_shape}"
            )

        self.client = voyageai.Client(
            api_key=api_key
        )

    def _load_chunks(
        self,
    ) -> list[dict]:

        chunks = []

        with CHUNKS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                chunks.append(
                    json.loads(line)
                )

        return chunks

    def _embed_query(
        self,
        query: str,
    ) -> np.ndarray:

        result = self.client.embed(
            [query],
            model=self.query_model,
            input_type="query",
            output_dimension=(
                self.dimension
            ),
        )

        vector = np.asarray(
            result.embeddings[0],
            dtype=np.float32,
        )

        norm = np.linalg.norm(
            vector
        )

        if norm <= 1e-12:
            raise RuntimeError(
                "Invalid zero query vector."
            )

        return vector / norm

    def _lexical_score(
        self,
        query: str,
        chunk: dict,
    ) -> float:

        query_tokens = tokenize(
            query
        )

        if not query_tokens:
            return 0.0

        searchable = " ".join(
            [
                chunk.get(
                    "text",
                    "",
                ),
                chunk.get(
                    "file_name",
                    "",
                ),
                chunk.get(
                    "source_path",
                    "",
                ),
            ]
        )

        chunk_tokens = tokenize(
            searchable
        )

        overlap = (
            query_tokens
            & chunk_tokens
        )

        return (
            len(overlap)
            / len(query_tokens)
        )

    def _canonical_source(
        self,
        chunk: dict,
    ) -> str:

        source = chunk.get(
            "source_path",
            "",
        )

        path = Path(source)

        # Treat PDF and DOCX versions of
        # the same Codex as one source family.
        return str(
            path.with_suffix("")
        ).replace(
            "\\",
            "/",
        ).lower()

    def _apply_diversity(
        self,
        results: list[
            SemanticSearchResult
        ],
        top_k: int,
    ) -> list[
        SemanticSearchResult
    ]:

        selected = []

        source_counts: dict[
            str,
            int
        ] = {}

        for result in results:

            source_key = (
                self._canonical_source(
                    result.chunk
                )
            )

            count = (
                source_counts.get(
                    source_key,
                    0,
                )
            )

            # Maximum two chunks from
            # the same canonical document.
            if count >= 2:
                continue

            selected.append(
                result
            )

            source_counts[
                source_key
            ] = count + 1

            if len(selected) >= top_k:
                break

        return selected

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 30,
        use_reranker: bool = True,
    ) -> list[
        SemanticSearchResult
    ]:

        query = query.strip()

        if not query:
            return []

        query_vector = (
            self._embed_query(
                query
            )
        )

        dense_scores = (
            self.embeddings
            @ query_vector
        )

        candidate_k = min(
            candidate_k,
            len(self.chunks),
        )

        candidate_indices = (
            np.argpartition(
                dense_scores,
                -candidate_k,
            )[
                -candidate_k:
            ]
        )

        candidates = []

        for raw_index in (
            candidate_indices
        ):

            index = int(
                raw_index
            )

            chunk = (
                self.chunks[
                    index
                ]
            )

            dense_score = float(
                dense_scores[
                    index
                ]
            )

            lexical_score = (
                self._lexical_score(
                    query,
                    chunk,
                )
            )

            local_score = (
                dense_score * 0.90
                + lexical_score * 0.10
            )

            candidates.append(
                SemanticSearchResult(
                    chunk=chunk,
                    dense_score=(
                        dense_score
                    ),
                    lexical_score=(
                        lexical_score
                    ),
                    rerank_score=None,
                    score=local_score,
                )
            )

        candidates.sort(
            key=lambda item:
                item.score,
            reverse=True,
        )

        if not use_reranker:

            return (
                self._apply_diversity(
                    candidates,
                    top_k,
                )
            )

        # ---------------------------------
        # Voyage cross-encoder reranking
        # ---------------------------------

        documents = []

        for item in candidates:

            chunk = item.chunk

            document = (
                f"Source: "
                f"{chunk.get('source_path', '')}\n"
                f"Document: "
                f"{chunk.get('file_name', '')}\n"
                f"Content:\n"
                f"{chunk.get('text', '')}"
            )

            documents.append(
                document
            )

        try:

            reranked = (
                self.client.rerank(
                    query=query,
                    documents=documents,
                    model=self.rerank_model,
                    top_k=min(
                        max(
                            top_k * 3,
                            10,
                        ),
                        len(documents),
                    ),
                )
            )

            final_results = []

            for ranked in (
                reranked.results
            ):

                item = candidates[
                    int(ranked.index)
                ]

                rerank_score = float(
                    ranked.relevance_score
                )

                # Reranker is dominant.
                # Lexical match still gives
                # exact archive names a small
                # deterministic advantage.
                final_score = (
                    rerank_score * 0.95
                    + item.lexical_score
                    * 0.05
                )

                final_results.append(
                    SemanticSearchResult(
                        chunk=item.chunk,
                        dense_score=(
                            item.dense_score
                        ),
                        lexical_score=(
                            item.lexical_score
                        ),
                        rerank_score=(
                            rerank_score
                        ),
                        score=(
                            final_score
                        ),
                    )
                )

            final_results.sort(
                key=lambda item:
                    item.score,
                reverse=True,
            )

            return (
                self._apply_diversity(
                    final_results,
                    top_k,
                )
            )

        except Exception as error:

            # Important for live demo:
            # Voyage reranker failure must
            # not destroy retrieval.
            print(
                "WARNING: Voyage reranker "
                "unavailable."
            )

            print(
                f"Reason: {error}"
            )

            print(
                "Falling back to dense + "
                "lexical retrieval."
            )

            return (
                self._apply_diversity(
                    candidates,
                    top_k,
                )
            )