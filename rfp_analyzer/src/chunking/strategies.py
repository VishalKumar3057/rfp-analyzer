"""Chunking strategies for RFP documents."""

import re
from abc import ABC, abstractmethod
from typing import Any

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from src.config import get_settings
from src.utils.logging import LoggerMixin


class ChunkingStrategy(ABC, LoggerMixin):
    """Abstract base class for chunking strategies."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize chunking strategy.

        Args:
            chunk_size: Target size for each chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split text into chunks.

        Args:
            text: Text to split.
            metadata: Optional metadata to attach to chunks.

        Returns:
            List of chunk dictionaries with content and metadata.
        """
        pass

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove page markers
        text = re.sub(r"\[Page \d+\]", "", text)
        return text.strip()


class RecursiveChunkingStrategy(ChunkingStrategy):
    """Recursive character-based chunking strategy."""

    SEPARATORS = [
        "\n\n\n",  # Multiple newlines (section breaks)
        "\n\n",     # Paragraph breaks
        "\n",       # Line breaks
        ". ",       # Sentences
        ", ",       # Clauses
        " ",        # Words
        "",         # Characters
    ]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split text using recursive character splitting."""
        cleaned_text = self._clean_text(text)
        docs = self._splitter.create_documents(
            [cleaned_text],
            metadatas=[metadata] if metadata else None,
        )
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata or {},
            }
            for doc in docs
        ]


class SemanticChunkingStrategy(ChunkingStrategy):
    """Semantic-based chunking using embeddings."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        breakpoint_threshold_type: str = "percentile",
    ):
        super().__init__(chunk_size, chunk_overlap)
        settings = get_settings()
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key.get_secret_value(),
        )
        self._breakpoint_threshold_type = breakpoint_threshold_type
        self._splitter: SemanticChunker | None = None

    def _get_splitter(self) -> SemanticChunker:
        """Get or create semantic chunker."""
        if self._splitter is None:
            self._splitter = SemanticChunker(
                embeddings=self._embeddings,
                breakpoint_threshold_type=self._breakpoint_threshold_type,
            )
        return self._splitter

    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split text using semantic similarity."""
        cleaned_text = self._clean_text(text)
        splitter = self._get_splitter()
        docs = splitter.create_documents([cleaned_text])
        
        return [
            {
                "content": doc.page_content,
                "metadata": {**(metadata or {}), "semantic_chunk": True},
            }
            for doc in docs
        ]


class HierarchicalChunkingStrategy(ChunkingStrategy):
    """Hierarchical chunking that preserves document structure."""

    SECTION_PATTERNS = [
        r"^(?:SECTION|Section|CHAPTER|Chapter)\s*(\d+(?:\.\d+)*)",
        r"^(\d+(?:\.\d+)*)\s+[A-Z]",
        r"^(?:ARTICLE|Article)\s+(\d+)",
    ]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split text preserving hierarchical structure."""
        cleaned_text = self._clean_text(text)
        sections = self._identify_sections(cleaned_text)
        
        if not sections:
            # Fall back to recursive splitting
            return self._fallback_split(cleaned_text, metadata)
        
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section, metadata)
            chunks.extend(section_chunks)

        return chunks

    def _identify_sections(self, text: str) -> list[dict[str, Any]]:
        """Identify sections in the text."""
        lines = text.split("\n")
        sections = []
        current_section = {"title": "Introduction", "number": "0", "content": [], "hierarchy": []}

        for line in lines:
            section_match = self._match_section(line)
            if section_match:
                if current_section["content"]:
                    current_section["content"] = " ".join(current_section["content"])
                    sections.append(current_section)
                current_section = {
                    "title": section_match.get("title", line.strip()),
                    "number": section_match.get("number", ""),
                    "content": [],
                    "hierarchy": self._build_hierarchy(section_match.get("number", "")),
                }
            else:
                current_section["content"].append(line)

        if current_section["content"]:
            current_section["content"] = " ".join(current_section["content"])
            sections.append(current_section)

        return sections

    def _match_section(self, line: str) -> dict[str, str] | None:
        """Match a line against section patterns."""
        line = line.strip()
        for pattern in self.SECTION_PATTERNS:
            match = re.match(pattern, line)
            if match:
                return {"number": match.group(1), "title": line}
        return None

    def _build_hierarchy(self, section_number: str) -> list[str]:
        """Build hierarchy from section number."""
        if not section_number:
            return []
        parts = section_number.split(".")
        return [".".join(parts[: i + 1]) for i in range(len(parts))]

    def _chunk_section(
        self, section: dict[str, Any], base_metadata: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Chunk a single section."""
        content = section.get("content", "")
        if not content or len(content) < 50:
            return []

        section_metadata = {
            **(base_metadata or {}),
            "section_title": section.get("title", ""),
            "section_number": section.get("number", ""),
            "section_hierarchy": section.get("hierarchy", []),
        }

        if len(content) <= self.chunk_size:
            return [{"content": content, "metadata": section_metadata}]

        # Use recursive splitter for large sections
        docs = self._fallback_splitter.create_documents(
            [content], metadatas=[section_metadata]
        )
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

    def _fallback_split(
        self, text: str, metadata: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Fallback to recursive splitting."""
        docs = self._fallback_splitter.create_documents(
            [text], metadatas=[metadata] if metadata else None
        )
        return [{"content": doc.page_content, "metadata": doc.metadata or {}} for doc in docs]
