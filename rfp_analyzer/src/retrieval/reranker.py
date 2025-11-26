"""Re-ranking module for improving retrieval quality."""

from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.utils.logging import LoggerMixin


class ReRanker(LoggerMixin):
    """Re-ranker for scoring and sorting retrieved documents."""

    RERANK_PROMPT = """You are a document relevance scorer. Given a query and a document, 
score how relevant the document is to the query on a scale of 0-100.

Query: {query}

Document:
{document}

Respond with ONLY a number between 0 and 100, where:
- 0-20: Not relevant
- 21-40: Slightly relevant
- 41-60: Moderately relevant
- 61-80: Highly relevant
- 81-100: Extremely relevant

Score:"""

    def __init__(self, model: str | None = None):
        """Initialize re-ranker.

        Args:
            model: LLM model for re-ranking.
        """
        settings = get_settings()
        self._llm = ChatOpenAI(
            model=model or settings.openai_model,
            temperature=0,
            openai_api_key=settings.openai_api_key.get_secret_value(),
        )

    def rerank(
        self,
        query: str,
        documents: list[tuple[Document, float]],
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Re-rank documents based on query relevance.

        Args:
            query: The search query.
            documents: List of (document, initial_score) tuples.
            top_k: Number of top results to return.

        Returns:
            Re-ranked list of (document, new_score) tuples.
        """
        if not documents:
            return []

        self.log_debug("Re-ranking documents", count=len(documents))

        reranked = []
        for doc, initial_score in documents:
            new_score = self._score_document(query, doc)
            # Combine initial and new scores
            combined_score = (initial_score * 0.3) + (new_score / 100 * 0.7)
            reranked.append((doc, combined_score))

        # Sort by score descending
        reranked.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def _score_document(self, query: str, document: Document) -> float:
        """Score a single document against the query.

        Args:
            query: Search query.
            document: Document to score.

        Returns:
            Relevance score (0-100).
        """
        try:
            prompt = self.RERANK_PROMPT.format(
                query=query,
                document=document.page_content[:2000],  # Limit content length
            )
            response = self._llm.invoke(prompt)
            score = float(response.content.strip())
            return min(100, max(0, score))
        except (ValueError, TypeError) as e:
            self.log_warning("Failed to parse rerank score", error=str(e))
            return 50.0  # Default middle score

    def batch_rerank(
        self,
        query: str,
        documents: list[tuple[Document, float]],
        batch_size: int = 5,
    ) -> list[tuple[Document, float]]:
        """Re-rank documents in batches for efficiency.

        Args:
            query: Search query.
            documents: Documents to rerank.
            batch_size: Size of each batch.

        Returns:
            Re-ranked documents.
        """
        if len(documents) <= batch_size:
            return self.rerank(query, documents)

        # Process in batches
        all_reranked = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            reranked_batch = self.rerank(query, batch)
            all_reranked.extend(reranked_batch)

        # Final sort
        all_reranked.sort(key=lambda x: x[1], reverse=True)
        return all_reranked


class SimpleReRanker(LoggerMixin):
    """Simple re-ranker using keyword matching (no LLM calls)."""

    def rerank(
        self,
        query: str,
        documents: list[tuple[Document, float]],
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Re-rank using keyword overlap scoring."""
        query_terms = set(query.lower().split())

        reranked = []
        for doc, initial_score in documents:
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
            # Combine scores
            combined_score = (initial_score * 0.5) + (overlap * 0.5)
            reranked.append((doc, combined_score))

        reranked.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked
