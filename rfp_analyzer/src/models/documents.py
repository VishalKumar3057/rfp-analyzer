"""Document-related Pydantic models."""

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for an RFP document."""

    source_file: str = Field(..., description="Original file path")
    project_name: str = Field(..., description="Project name extracted from document")
    document_type: str = Field(default="RFP", description="Type of document")
    total_pages: int = Field(default=0, ge=0, description="Total number of pages")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    file_size_bytes: int = Field(default=0, ge=0)
    
    # Extracted structure
    sections: list[str] = Field(default_factory=list, description="List of section titles")
    has_appendices: bool = Field(default=False)
    
    model_config = {"frozen": False}


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    document_id: UUID = Field(..., description="Parent document ID")
    chunk_id: UUID = Field(default_factory=uuid4)
    chunk_index: int = Field(..., ge=0, description="Index of chunk in document")
    
    # Location information
    page_numbers: list[int] = Field(default_factory=list, description="Pages this chunk spans")
    section_title: str | None = Field(default=None, description="Section this chunk belongs to")
    section_hierarchy: list[str] = Field(default_factory=list, description="Hierarchy path")
    
    # Content classification
    content_type: str = Field(default="text", description="Type: text, table, list, header")
    contains_requirements: bool = Field(default=False)
    requirement_ids: list[str] = Field(default_factory=list)
    
    # Cross-references
    references_sections: list[str] = Field(default_factory=list)
    referenced_by: list[str] = Field(default_factory=list)
    
    # For filtering
    project_name: str = Field(default="", description="Project name for filtering")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")
    
    model_config = {"frozen": False}


class DocumentChunk(BaseModel):
    """A chunk of document content with metadata."""

    content: str = Field(..., description="The text content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    
    # Embedding placeholder
    embedding: list[float] | None = Field(default=None, exclude=True)
    
    def to_langchain_document(self) -> dict[str, Any]:
        """Convert to LangChain document format."""
        return {
            "page_content": self.content,
            "metadata": self.metadata.model_dump(mode="json"),
        }
    
    model_config = {"frozen": False}


class RFPDocument(BaseModel):
    """Complete RFP document representation."""

    id: UUID = Field(default_factory=uuid4)
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    chunks: list[DocumentChunk] = Field(default_factory=list)
    raw_content: str = Field(default="", description="Full raw text content")
    
    @property
    def chunk_count(self) -> int:
        """Get the number of chunks."""
        return len(self.chunks)
    
    @classmethod
    def from_file(cls, file_path: Path, project_name: str) -> "RFPDocument":
        """Create a document placeholder from file path."""
        return cls(
            metadata=DocumentMetadata(
                source_file=str(file_path),
                project_name=project_name,
                file_size_bytes=file_path.stat().st_size if file_path.exists() else 0,
            )
        )
    
    model_config = {"frozen": False}

