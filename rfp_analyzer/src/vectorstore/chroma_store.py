"""ChromaDB vector store implementation."""

from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import get_settings
from src.models.documents import DocumentChunk
from src.vectorstore.embeddings import EmbeddingService
from src.utils.logging import LoggerMixin


class ChromaVectorStore(LoggerMixin):
    """Vector store using ChromaDB with LangChain integration."""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: Path | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        """Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection.
            persist_directory: Directory for persistence.
            embedding_service: Optional embedding service instance.
        """
        settings = get_settings()
        
        self._collection_name = collection_name or settings.collection_name
        self._persist_directory = persist_directory or settings.chroma_persist_directory
        self._embedding_service = embedding_service or EmbeddingService()
        
        # Ensure directory exists
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma
        self._vectorstore = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embedding_service.embeddings,
            persist_directory=str(self._persist_directory),
        )
        
        self.log_info(
            "ChromaDB initialized",
            collection=self._collection_name,
            persist_dir=str(self._persist_directory),
        )

    @property
    def vectorstore(self) -> Chroma:
        """Get the underlying Chroma vectorstore."""
        return self._vectorstore

    def add_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        """Add document chunks to the vector store.

        Args:
            chunks: List of document chunks to add.

        Returns:
            List of chunk IDs.
        """
        if not chunks:
            return []
        
        documents = []
        ids = []
        
        for chunk in chunks:
            doc = Document(
                page_content=chunk.content,
                metadata=self._prepare_metadata(chunk.metadata.model_dump(mode="json")),
            )
            documents.append(doc)
            ids.append(str(chunk.metadata.chunk_id))
        
        self._vectorstore.add_documents(documents, ids=ids)
        
        self.log_info("Added chunks to vector store", count=len(chunks))
        return ids

    def _prepare_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Prepare metadata for ChromaDB (flatten complex types)."""
        prepared = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings for Chroma
                prepared[key] = ",".join(str(v) for v in value) if value else ""
            elif isinstance(value, UUID):
                prepared[key] = str(value)
            elif value is None:
                prepared[key] = ""
            else:
                prepared[key] = str(value)
        return prepared

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter_dict: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents.

        Args:
            query: Search query.
            k: Number of results.
            filter_dict: Optional metadata filter.
            score_threshold: Minimum similarity score (0-1).

        Returns:
            List of (document, score) tuples with normalized scores (0-1).
        """
        self.log_debug("Similarity search", query=query[:50], k=k)

        # Use similarity_search_with_score which returns distance (lower is better)
        results = self._vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict,
        )

        # Normalize scores: convert distance to similarity (0-1 range)
        # Chroma uses L2 distance, so we convert: similarity = 1 / (1 + distance)
        normalized_results = []
        for doc, distance in results:
            # Convert distance to similarity score (0-1)
            similarity = 1 / (1 + distance)
            normalized_results.append((doc, similarity))

        if score_threshold is not None:
            normalized_results = [
                (doc, score) for doc, score in normalized_results
                if score >= score_threshold
            ]

        return normalized_results

    def search_by_project(
        self,
        query: str,
        project_name: str,
        k: int = 10,
    ) -> list[tuple[Document, float]]:
        """Search within a specific project.

        Args:
            query: Search query.
            project_name: Project name to filter by.
            k: Number of results.

        Returns:
            List of (document, score) tuples.
        """
        return self.similarity_search(
            query=query,
            k=k,
            filter_dict={"project_name": project_name},
        )

    def get_retriever(
        self,
        search_type: str = "similarity",
        k: int = 10,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
    ):
        """Get a LangChain retriever.

        Args:
            search_type: Type of search (similarity, mmr).
            k: Number of results.
            score_threshold: Minimum score threshold.
            filter_dict: Metadata filter.

        Returns:
            LangChain retriever.
        """
        search_kwargs: dict[str, Any] = {"k": k}

        if filter_dict:
            search_kwargs["filter"] = filter_dict

        if score_threshold is not None and search_type == "similarity":
            search_kwargs["score_threshold"] = score_threshold
            return self._vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=search_kwargs,
            )

        return self._vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all chunks for a document.

        Args:
            document_id: Document ID to delete chunks for.

        Returns:
            True if successful.
        """
        try:
            # Get all chunk IDs for this document
            results = self._vectorstore.get(
                where={"document_id": document_id},
            )
            if results and results.get("ids"):
                self._vectorstore.delete(ids=results["ids"])
                self.log_info(
                    "Deleted chunks for document",
                    document_id=document_id,
                    count=len(results["ids"]),
                )
            return True
        except Exception as e:
            self.log_error("Failed to delete chunks", error=str(e))
            return False

    def get_collection_stats(self) -> dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary with collection stats.
        """
        try:
            collection = self._vectorstore._collection
            count = collection.count()
            return {
                "collection_name": self._collection_name,
                "document_count": count,
                "persist_directory": str(self._persist_directory),
            }
        except Exception as e:
            self.log_error("Failed to get stats", error=str(e))
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """Clear all documents from the collection.

        Returns:
            True if successful.
        """
        try:
            # Delete and recreate collection
            self._vectorstore.delete_collection()
            self._vectorstore = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedding_service.embeddings,
                persist_directory=str(self._persist_directory),
            )
            self.log_info("Collection cleared")
            return True
        except Exception as e:
            self.log_error("Failed to clear collection", error=str(e))
            return False
