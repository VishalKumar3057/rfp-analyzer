"""Intelligent chunker that combines multiple strategies."""

import re
from typing import Any
from uuid import uuid4

from src.chunking.strategies import (
    ChunkingStrategy,
    HierarchicalChunkingStrategy,
    RecursiveChunkingStrategy,
)
from src.config import get_settings
from src.models.documents import ChunkMetadata, DocumentChunk, RFPDocument
from src.utils.logging import LoggerMixin


class IntelligentChunker(LoggerMixin):
    """Intelligent document chunker that handles RFP-specific patterns."""

    # Patterns for identifying requirements
    REQUIREMENT_PATTERNS = [
        r"(?:shall|must|will|should|required to)\s+",
        r"(?:requirement|req\.?)\s*[:#]?\s*\d+",
        r"(?:mandatory|optional|critical)\s+requirement",
        r"the\s+(?:system|solution|vendor|contractor)\s+(?:shall|must|will)",
    ]

    # Patterns for cross-references
    CROSS_REF_PATTERNS = [
        r"(?:see|refer to|as described in|per)\s+(?:section|appendix|article)\s+[\d\w\.]+",
        r"(?:section|appendix|article)\s+[\d\w\.]+\s+(?:describes|specifies|defines)",
        r"in accordance with\s+[\w\s]+",
    ]

    # Keywords for categorization
    CATEGORY_KEYWORDS = {
        "technical": ["system", "software", "hardware", "architecture", "integration"],
        "security": ["security", "authentication", "encryption", "access control", "audit"],
        "performance": ["performance", "latency", "throughput", "scalability", "availability"],
        "compliance": ["compliance", "regulation", "standard", "certification", "audit"],
        "timeline": ["timeline", "schedule", "deadline", "milestone", "delivery"],
        "budget": ["budget", "cost", "pricing", "payment", "financial"],
    }

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        use_hierarchical: bool = True,
    ):
        """Initialize the intelligent chunker.

        Args:
            chunk_size: Size of chunks.
            chunk_overlap: Overlap between chunks.
            use_hierarchical: Whether to use hierarchical chunking.
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        if use_hierarchical:
            self._strategy: ChunkingStrategy = HierarchicalChunkingStrategy(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            self._strategy = RecursiveChunkingStrategy(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

    def chunk_document(self, document: RFPDocument) -> list[DocumentChunk]:
        """Chunk an RFP document into intelligent chunks.

        Args:
            document: RFP document to chunk.

        Returns:
            List of document chunks with metadata.
        """
        self.log_info(
            "Chunking document",
            document_id=str(document.id),
            content_length=len(document.raw_content),
        )

        base_metadata = {
            "document_id": str(document.id),
            "project_name": document.metadata.project_name,
            "source_file": document.metadata.source_file,
        }

        # Split using strategy
        raw_chunks = self._strategy.split(document.raw_content, base_metadata)
        
        # Enhance chunks with intelligent metadata
        enhanced_chunks = []
        for idx, chunk_data in enumerate(raw_chunks):
            enhanced_chunk = self._enhance_chunk(chunk_data, idx, document)
            enhanced_chunks.append(enhanced_chunk)

        self.log_info(
            "Document chunked successfully",
            chunk_count=len(enhanced_chunks),
        )
        
        return enhanced_chunks

    def _enhance_chunk(
        self, chunk_data: dict[str, Any], index: int, document: RFPDocument
    ) -> DocumentChunk:
        """Enhance a chunk with intelligent metadata."""
        content = chunk_data["content"]
        base_meta = chunk_data.get("metadata", {})
        
        # Extract page numbers from content markers
        page_numbers = self._extract_page_numbers(content)
        
        # Detect if chunk contains requirements
        contains_requirements, requirement_ids = self._detect_requirements(content)
        
        # Find cross-references
        references = self._find_cross_references(content)
        
        # Categorize content
        keywords = self._extract_keywords(content)
        
        chunk_metadata = ChunkMetadata(
            document_id=document.id,
            chunk_id=uuid4(),
            chunk_index=index,
            page_numbers=page_numbers,
            section_title=base_meta.get("section_title"),
            section_hierarchy=base_meta.get("section_hierarchy", []),
            content_type=self._detect_content_type(content),
            contains_requirements=contains_requirements,
            requirement_ids=requirement_ids,
            references_sections=references,
            project_name=document.metadata.project_name,
            keywords=keywords,
        )

        return DocumentChunk(content=content, metadata=chunk_metadata)

    def _extract_page_numbers(self, content: str) -> list[int]:
        """Extract page numbers from content."""
        matches = re.findall(r"\[Page (\d+)\]", content)
        return list(set(int(m) for m in matches))

    def _detect_requirements(self, content: str) -> tuple[bool, list[str]]:
        """Detect if content contains requirements."""
        content_lower = content.lower()
        has_requirements = any(
            re.search(pattern, content_lower) for pattern in self.REQUIREMENT_PATTERNS
        )
        
        # Extract requirement IDs
        req_ids = re.findall(r"(?:REQ|Req|req)[.\-_]?\s*(\d+(?:\.\d+)*)", content)

        return has_requirements, req_ids

    def _find_cross_references(self, content: str) -> list[str]:
        """Find cross-references to other sections."""
        references = []
        content_lower = content.lower()

        for pattern in self.CROSS_REF_PATTERNS:
            matches = re.findall(pattern, content_lower)
            references.extend(matches)

        # Extract specific section references
        section_refs = re.findall(
            r"(?:section|appendix|article)\s+([\d\w\.]+)", content_lower
        )
        references.extend(section_refs)

        return list(set(references))

    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content in the chunk."""
        # Check for tables
        if re.search(r"[\|\+\-]{3,}", content) or content.count("\t") > 5:
            return "table"

        # Check for lists
        list_patterns = [
            r"^\s*[\-\*\•]\s+",
            r"^\s*\d+[\.\)]\s+",
            r"^\s*[a-z][\.\)]\s+",
        ]
        list_count = sum(
            len(re.findall(pattern, content, re.MULTILINE))
            for pattern in list_patterns
        )
        if list_count > 3:
            return "list"

        # Check for headers
        if len(content) < 200 and content.isupper():
            return "header"

        return "text"

    def _extract_keywords(self, content: str) -> list[str]:
        """Extract keywords for categorization."""
        content_lower = content.lower()
        keywords = []

        for category, terms in self.CATEGORY_KEYWORDS.items():
            if any(term in content_lower for term in terms):
                keywords.append(category)

        return keywords

    def chunk_documents(self, documents: list[RFPDocument]) -> list[DocumentChunk]:
        """Chunk multiple documents.

        Args:
            documents: List of RFP documents.

        Returns:
            Combined list of all chunks.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        self.log_info(
            "All documents chunked",
            document_count=len(documents),
            total_chunks=len(all_chunks),
        )

        return all_chunks
