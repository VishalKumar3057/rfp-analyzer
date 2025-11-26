"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from src.models.requests import AnalysisRequest, QueryType
from src.models.responses import AnalysisResponse, ExtractedRequirement, RetrievedChunk
from src.models.documents import DocumentMetadata, ChunkMetadata, DocumentChunk


class TestAnalysisRequest:
    """Tests for AnalysisRequest model."""

    def test_valid_request(self):
        """Test creating a valid request."""
        request = AnalysisRequest(
            query="What are the security requirements?",
            project_name="Test Project",
            query_type=QueryType.REQUIREMENT_EXTRACTION,
        )
        assert request.query == "What are the security requirements?"
        assert request.project_name == "Test Project"
        assert request.query_type == QueryType.REQUIREMENT_EXTRACTION

    def test_default_query_type(self):
        """Test default query type is GENERAL."""
        request = AnalysisRequest(query="Test query here")
        assert request.query_type == QueryType.GENERAL

    def test_query_too_short(self):
        """Test validation for short queries."""
        with pytest.raises(ValidationError):
            AnalysisRequest(query="Hi")

    def test_additional_context(self):
        """Test additional context field."""
        request = AnalysisRequest(
            query="Check compliance",
            additional_context={"approach": "OAuth 2.0"},
        )
        assert request.additional_context["approach"] == "OAuth 2.0"


class TestAnalysisResponse:
    """Tests for AnalysisResponse model."""

    def test_valid_response(self):
        """Test creating a valid response."""
        response = AnalysisResponse(
            extracted_requirements=[],
            reasoning="Test reasoning",
            confidence=85,
        )
        assert response.reasoning == "Test reasoning"
        assert response.confidence == 85

    def test_with_requirements(self):
        """Test response with requirements."""
        req = ExtractedRequirement(
            requirement_id="REQ-001",
            title="Test Requirement",
            description="Description here",
        )
        response = AnalysisResponse(
            extracted_requirements=[req],
            reasoning="Found one requirement",
            confidence=90,
        )
        assert len(response.extracted_requirements) == 1
        assert response.extracted_requirements[0].requirement_id == "REQ-001"

    def test_confidence_bounds(self):
        """Test confidence score bounds."""
        # Valid confidence
        response = AnalysisResponse(
            extracted_requirements=[],
            reasoning="Test",
            confidence=100,
        )
        assert response.confidence == 100

        # Confidence should be clamped or validated
        response2 = AnalysisResponse(
            extracted_requirements=[],
            reasoning="Test",
            confidence=0,
        )
        assert response2.confidence == 0


class TestExtractedRequirement:
    """Tests for ExtractedRequirement model."""

    def test_minimal_requirement(self):
        """Test creating requirement with minimal fields."""
        req = ExtractedRequirement(
            requirement_id="REQ-001",
            title="Test",
            description="Description",
        )
        assert req.requirement_id == "REQ-001"
        assert req.section is None
        assert req.priority is None

    def test_full_requirement(self):
        """Test creating requirement with all fields."""
        req = ExtractedRequirement(
            requirement_id="REQ-001",
            title="Security Requirement",
            description="Must implement MFA",
            section="1.1 Security",
            page_number=5,
            priority="high",
            category="security",
            related_requirements=["REQ-002"],
        )
        assert req.priority == "high"
        assert req.category == "security"
        assert "REQ-002" in req.related_requirements


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_valid_metadata(self):
        """Test creating valid document metadata."""
        meta = DocumentMetadata(
            source_file="test.pdf",
            project_name="Test Project",
            total_pages=10,
        )
        assert meta.source_file == "test.pdf"
        assert meta.total_pages == 10

    def test_optional_fields(self):
        """Test optional metadata fields."""
        meta = DocumentMetadata(
            source_file="test.pdf",
            project_name="Test",
        )
        assert meta.total_pages is None
        assert meta.sections == []


class TestRetrievedChunk:
    """Tests for RetrievedChunk model."""

    def test_valid_chunk(self):
        """Test creating a valid retrieved chunk."""
        chunk = RetrievedChunk(
            chunk_id="chunk-001",
            content="Test content here",
            relevance_score=0.85,
            source_document="test.pdf",
        )
        assert chunk.relevance_score == 0.85
        assert chunk.source_document == "test.pdf"

