"""Multi-stage retrieval pipeline."""

from typing import Any

from langchain_core.documents import Document

from src.config import get_settings
from src.models.responses import RetrievedChunk
from src.retrieval.context_enricher import ContextEnricher
from src.retrieval.reranker import ReRanker, SimpleReRanker
from src.vectorstore.chroma_store import ChromaVectorStore
from src.utils.logging import LoggerMixin


class RetrievalPipeline(LoggerMixin):
    """Multi-stage retrieval pipeline for RFP analysis.

    Stages:
    1. Initial retrieval: Semantic search with embeddings
    2. Filtering: Use metadata to refine results
    3. Re-ranking: Score chunks by relevance
    4. Context enrichment: Include related sections
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        use_llm_reranking: bool = True,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ):
        """Initialize retrieval pipeline.

        Args:
            vector_store: Vector store for retrieval.
            use_llm_reranking: Whether to use LLM for re-ranking.
            top_k: Number of results to return.
            similarity_threshold: Minimum similarity score.
        """
        settings = get_settings()
        
        self._vector_store = vector_store
        self._top_k = top_k or settings.top_k_results
        self._similarity_threshold = similarity_threshold or settings.similarity_threshold
        
        # Initialize components
        self._reranker = ReRanker() if use_llm_reranking else SimpleReRanker()
        self._context_enricher = ContextEnricher(vector_store)

    def retrieve(
        self,
        query: str,
        project_name: str | None = None,
        section_filter: str | None = None,
        include_enrichment: bool = True,
    ) -> list[RetrievedChunk]:
        """Execute the full retrieval pipeline.

        Args:
            query: User query.
            project_name: Optional project filter.
            section_filter: Optional section filter.
            include_enrichment: Whether to include context enrichment.

        Returns:
            List of retrieved chunks with relevance info.
        """
        self.log_info(
            "Starting retrieval pipeline",
            query=query[:100],
            project=project_name,
        )

        # Stage 1: Initial semantic search
        initial_results = self._initial_retrieval(
            query=query,
            project_name=project_name,
            k=self._top_k * 2,  # Get more for filtering
        )

        if not initial_results:
            self.log_warning("No initial results found")
            return []

        # Stage 2: Metadata filtering
        filtered_results = self._apply_filters(
            documents=initial_results,
            section_filter=section_filter,
        )

        # Stage 3: Re-ranking
        reranked_results = self._reranker.rerank(
            query=query,
            documents=filtered_results,
            top_k=self._top_k,
        )

        # Stage 4: Context enrichment
        if include_enrichment:
            enriched_results = self._context_enricher.enrich(
                documents=reranked_results,
                query=query,
                max_additional=3,
            )
        else:
            enriched_results = reranked_results

        # Convert to response format
        retrieved_chunks = self._to_retrieved_chunks(enriched_results)

        self.log_info(
            "Retrieval complete",
            initial=len(initial_results),
            filtered=len(filtered_results),
            final=len(retrieved_chunks),
        )

        return retrieved_chunks

    def _initial_retrieval(
        self,
        query: str,
        project_name: str | None,
        k: int,
    ) -> list[tuple[Document, float]]:
        """Stage 1: Initial semantic search."""
        if project_name:
            return self._vector_store.search_by_project(
                query=query,
                project_name=project_name,
                k=k,
            )
        return self._vector_store.similarity_search(
            query=query,
            k=k,
            score_threshold=self._similarity_threshold,
        )

    def _apply_filters(
        self,
        documents: list[tuple[Document, float]],
        section_filter: str | None,
    ) -> list[tuple[Document, float]]:
        """Stage 2: Apply metadata filters."""
        if not section_filter:
            return documents

        filtered = []
        section_lower = section_filter.lower()
        
        for doc, score in documents:
            section_title = doc.metadata.get("section_title", "").lower()
            section_hierarchy = doc.metadata.get("section_hierarchy", "").lower()
            
            if section_lower in section_title or section_lower in section_hierarchy:
                filtered.append((doc, score))

        return filtered if filtered else documents  # Fallback to all if no matches

    def _to_retrieved_chunks(
        self,
        documents: list[tuple[Document, float]],
    ) -> list[RetrievedChunk]:
        """Convert documents to RetrievedChunk format."""
        chunks = []

        for doc, score in documents:
            page_numbers_str = doc.metadata.get("page_numbers", "")
            page_numbers = []
            if page_numbers_str:
                try:
                    page_numbers = [int(p) for p in str(page_numbers_str).split(",") if p]
                except (ValueError, AttributeError):
                    pass

            chunk = RetrievedChunk(
                chunk_id=doc.metadata.get("chunk_id", ""),
                content=doc.page_content,
                relevance_score=float(score),
                source_document=doc.metadata.get("source_file", ""),
                section=doc.metadata.get("section_title"),
                page_numbers=page_numbers,
            )
            chunks.append(chunk)

        return chunks

    def retrieve_for_requirements(
        self,
        subsystem: str,
        project_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Specialized retrieval for requirement extraction.

        Args:
            subsystem: Target subsystem.
            project_name: Optional project filter.

        Returns:
            Retrieved requirement chunks.
        """
        query = f"requirements specifications for {subsystem} subsystem technical functional"
        return self.retrieve(
            query=query,
            project_name=project_name,
            include_enrichment=True,
        )

    def retrieve_for_compliance(
        self,
        approach_description: str,
        section: str | None = None,
        project_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Specialized retrieval for compliance checking.

        Args:
            approach_description: Description of the approach.
            section: Target section.
            project_name: Optional project filter.

        Returns:
            Retrieved chunks for compliance checking.
        """
        query = f"requirements compliance criteria for {approach_description}"
        return self.retrieve(
            query=query,
            project_name=project_name,
            section_filter=section,
            include_enrichment=True,
        )

    def get_retriever(self, project_name: str | None = None):
        """Get a LangChain-compatible retriever.

        Args:
            project_name: Optional project filter.

        Returns:
            LangChain retriever.
        """
        filter_dict = {"project_name": project_name} if project_name else None
        return self._vector_store.get_retriever(
            search_type="similarity",
            k=self._top_k,
            filter_dict=filter_dict,
        )
