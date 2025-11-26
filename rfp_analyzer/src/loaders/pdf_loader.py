"""PDF document loader using PyMuPDF for robust extraction."""

import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import fitz  # PyMuPDF

from src.loaders.base import BaseDocumentLoader
from src.models.documents import DocumentMetadata, RFPDocument


class PDFLoader(BaseDocumentLoader):
    """PDF document loader with advanced extraction capabilities."""

    SECTION_PATTERNS = [
        r"^(?:SECTION|Section|CHAPTER|Chapter)\s*(\d+(?:\.\d+)*)\s*[:\-\.]?\s*(.+)$",
        r"^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s]+)$",
        r"^([A-Z]{1,3}(?:\.\d+)*)\s*[:\-\.]?\s*(.+)$",
        r"^((?:ARTICLE|Article)\s+\d+)\s*[:\-\.]?\s*(.+)$",
    ]

    def __init__(self, file_path: Path, project_name: str | None = None):
        """Initialize PDF loader.

        Args:
            file_path: Path to PDF file.
            project_name: Optional project name override.
        """
        super().__init__(file_path)
        self._project_name = project_name
        self._doc: fitz.Document | None = None

    @property
    def supported_extensions(self) -> list[str]:
        """Supported file extensions."""
        return [".pdf", ".PDF"]

    def _open_document(self) -> fitz.Document:
        """Open the PDF document."""
        if self._doc is None:
            self._doc = fitz.open(self.file_path)
        return self._doc

    def _close_document(self) -> None:
        """Close the PDF document."""
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def load(self) -> RFPDocument:
        """Load and parse the PDF document."""
        self.log_info("Loading PDF document", file=str(self.file_path))
        
        try:
            doc = self._open_document()
            metadata = self.extract_metadata()
            sections = self.extract_sections()
            raw_content = self._extract_full_text(doc)
            
            rfp_doc = RFPDocument(
                id=uuid4(),
                metadata=DocumentMetadata(
                    source_file=str(self.file_path),
                    project_name=self.get_project_name(),
                    total_pages=len(doc),
                    file_size_bytes=self.file_path.stat().st_size,
                    sections=[s["title"] for s in sections],
                    has_appendices=self._has_appendices(sections),
                ),
                raw_content=raw_content,
            )
            
            self.log_info(
                "PDF loaded successfully",
                pages=len(doc),
                sections=len(sections),
            )
            return rfp_doc
            
        finally:
            self._close_document()

    def _extract_full_text(self, doc: fitz.Document) -> str:
        """Extract full text from all pages."""
        text_parts = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            text_parts.append(f"[Page {page_num + 1}]\n{text}")
        return "\n\n".join(text_parts)

    def extract_metadata(self) -> dict[str, Any]:
        """Extract PDF metadata."""
        doc = self._open_document()
        pdf_metadata = doc.metadata or {}
        
        return {
            "title": pdf_metadata.get("title", ""),
            "author": pdf_metadata.get("author", ""),
            "subject": pdf_metadata.get("subject", ""),
            "creator": pdf_metadata.get("creator", ""),
            "producer": pdf_metadata.get("producer", ""),
            "creation_date": pdf_metadata.get("creationDate", ""),
            "modification_date": pdf_metadata.get("modDate", ""),
            "page_count": len(doc),
        }

    def extract_sections(self) -> list[dict[str, Any]]:
        """Extract document sections with hierarchy."""
        doc = self._open_document()
        sections = []
        current_section = None
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") != 0:  # Text block
                    continue
                    
                for line in block.get("lines", []):
                    text = "".join(span["text"] for span in line.get("spans", []))
                    text = text.strip()
                    
                    if not text:
                        continue
                    
                    section_match = self._match_section_header(text, line.get("spans", []))
                    if section_match:
                        if current_section:
                            sections.append(current_section)
                        current_section = {
                            "title": section_match["title"],
                            "number": section_match["number"],
                            "page": page_num + 1,
                            "content": [],
                            "hierarchy": self._parse_hierarchy(section_match["number"]),
                        }
                    elif current_section:
                        current_section["content"].append(text)
        
        if current_section:
            sections.append(current_section)
        
        # Join content for each section
        for section in sections:
            section["content"] = " ".join(section["content"])

        return sections

    def _match_section_header(
        self, text: str, spans: list[dict[str, Any]]
    ) -> dict[str, str] | None:
        """Match text against section header patterns."""
        # Check if text looks like a header (often bold or larger font)
        is_potential_header = False
        if spans:
            avg_size = sum(s.get("size", 12) for s in spans) / len(spans)
            is_bold = any("bold" in s.get("font", "").lower() for s in spans)
            is_potential_header = is_bold or avg_size > 12

        for pattern in self.SECTION_PATTERNS:
            match = re.match(pattern, text, re.MULTILINE)
            if match:
                return {
                    "number": match.group(1),
                    "title": match.group(2).strip() if match.lastindex >= 2 else text,
                }

        # Check for all-caps headers
        if is_potential_header and text.isupper() and len(text) < 100:
            return {"number": "", "title": text}

        return None

    def _parse_hierarchy(self, section_number: str) -> list[str]:
        """Parse section number into hierarchy levels."""
        if not section_number:
            return []
        parts = re.split(r"[.\-]", section_number)
        hierarchy = []
        for i in range(len(parts)):
            hierarchy.append(".".join(parts[: i + 1]))
        return hierarchy

    def _has_appendices(self, sections: list[dict[str, Any]]) -> bool:
        """Check if document has appendices."""
        appendix_patterns = ["appendix", "annex", "attachment", "exhibit"]
        for section in sections:
            title_lower = section.get("title", "").lower()
            if any(pattern in title_lower for pattern in appendix_patterns):
                return True
        return False

    def get_project_name(self) -> str:
        """Extract project name from document or filename."""
        if self._project_name:
            return self._project_name

        # Try to extract from PDF metadata
        try:
            doc = self._open_document()
            title = doc.metadata.get("title", "")
            if title and len(title) > 3:
                return title
        except Exception:
            pass

        # Fall back to filename
        return super().get_project_name()

