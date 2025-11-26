"""Context enricher for including related sections."""

import re
from typing import Any

from langchain_core.documents import Document

from src.vectorstore.chroma_store import ChromaVectorStore
from src.utils.logging import LoggerMixin


class ContextEnricher(LoggerMixin):
    """Enriches retrieved context with related sections."""

    CROSS_REF_PATTERNS = [
        r"(?:see|refer to|as described in|per)\s+(?:section|appendix|article)\s+([\d\w\.]+)",
        r"(?:section|appendix|article)\s+([\d\w\.]+)\s+(?:describes|specifies|defines|contains)",
        r"in accordance with\s+(?:section|appendix)?\s*([\d\w\.]+)",
    ]

    def __init__(self, vector_store: ChromaVectorStore):
        """Initialize context enricher.

        Args:
            vector_store: Vector store for retrieving related sections.
        """
        self._vector_store = vector_store

    def enrich(
        self,
        documents: list[tuple[Document, float]],
        query: str,
        max_additional: int = 3,
    ) -> list[tuple[Document, float]]:
        """Enrich documents with related sections.

        Args:
            documents: Retrieved documents with scores.
            query: Original query.
            max_additional: Maximum additional documents to add.

        Returns:
            Enriched list of documents.
        """
        if not documents:
            return []

        self.log_debug("Enriching context", original_count=len(documents))

        # Extract cross-references from retrieved documents
        referenced_sections = self._extract_references(documents)

        # Get existing document IDs to avoid duplicates
        existing_ids = {
            doc.metadata.get("chunk_id") for doc, _ in documents
        }

        additional_docs = []
        for section_ref in referenced_sections[:max_additional]:
            related = self._find_related_section(section_ref, existing_ids)
            if related:
                additional_docs.extend(related)
                existing_ids.update(
                    doc.metadata.get("chunk_id") for doc, _ in related
                )

        # Add related documents with slightly lower scores
        for doc, score in additional_docs:
            adjusted_score = score * 0.8  # Reduce score for related docs
            documents.append((doc, adjusted_score))

        self.log_debug(
            "Context enriched",
            original=len(documents) - len(additional_docs),
            added=len(additional_docs),
        )

        return documents

    def _extract_references(
        self, documents: list[tuple[Document, float]]
    ) -> list[str]:
        """Extract cross-references from documents."""
        references = []
        
        for doc, _ in documents:
            content = doc.page_content.lower()
            
            for pattern in self.CROSS_REF_PATTERNS:
                matches = re.findall(pattern, content)
                references.extend(matches)

        # Also check metadata for stored references
        for doc, _ in documents:
            refs = doc.metadata.get("references_sections", "")
            if refs:
                references.extend(refs.split(","))

        # Deduplicate and clean
        unique_refs = list(set(ref.strip() for ref in references if ref.strip()))
        return unique_refs

    def _find_related_section(
        self,
        section_ref: str,
        exclude_ids: set[str],
    ) -> list[tuple[Document, float]]:
        """Find documents related to a section reference."""
        try:
            # Search for the referenced section
            results = self._vector_store.similarity_search(
                query=f"section {section_ref}",
                k=2,
            )

            # Filter out already included documents
            filtered = [
                (doc, score)
                for doc, score in results
                if doc.metadata.get("chunk_id") not in exclude_ids
            ]

            return filtered

        except Exception as e:
            self.log_warning(
                "Failed to find related section",
                section=section_ref,
                error=str(e),
            )
            return []

    def get_parent_context(
        self,
        document: Document,
    ) -> Document | None:
        """Get parent section context for a document."""
        hierarchy = document.metadata.get("section_hierarchy", "")
        if not hierarchy:
            return None

        # Parse hierarchy and find parent
        sections = hierarchy.split(",") if isinstance(hierarchy, str) else hierarchy
        if len(sections) < 2:
            return None

        parent_section = sections[-2]  # Second to last in hierarchy

        try:
            results = self._vector_store.similarity_search(
                query=f"section {parent_section}",
                k=1,
            )
            if results:
                return results[0][0]
        except Exception as e:
            self.log_warning("Failed to get parent context", error=str(e))

        return None
