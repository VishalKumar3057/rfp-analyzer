"""Embedding service using OpenAI."""

from typing import Any

from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.utils.logging import LoggerMixin


class EmbeddingService(LoggerMixin):
    """Service for generating embeddings using OpenAI."""

    def __init__(
        self,
        model: str | None = None,
        dimensions: int | None = None,
    ):
        """Initialize embedding service.

        Args:
            model: OpenAI embedding model name.
            dimensions: Optional dimension reduction.
        """
        settings = get_settings()
        self._model = model or settings.openai_embedding_model
        self._dimensions = dimensions
        
        self._embeddings = OpenAIEmbeddings(
            model=self._model,
            openai_api_key=settings.openai_api_key.get_secret_value(),
            dimensions=dimensions,
        )
        
        self.log_info(
            "Embedding service initialized",
            model=self._model,
            dimensions=dimensions,
        )

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Get the LangChain embeddings instance."""
        return self._embeddings

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        return self._embeddings.embed_query(text)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        self.log_debug("Embedding texts", count=len(texts))
        return self._embeddings.embed_documents(texts)

    def embed_with_metadata(
        self, texts: list[str], metadatas: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Embed texts with their metadata.

        Args:
            texts: List of texts.
            metadatas: List of metadata dictionaries.

        Returns:
            List of dicts with text, embedding, and metadata.
        """
        embeddings = self.embed_texts(texts)
        
        return [
            {
                "text": text,
                "embedding": embedding,
                "metadata": metadata,
            }
            for text, embedding, metadata in zip(texts, embeddings, metadatas)
        ]

