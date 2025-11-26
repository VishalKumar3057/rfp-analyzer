"""Response models for RFP Analyzer."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ExtractedRequirement(BaseModel):
    """A single extracted requirement."""

    requirement_id: str = Field(..., description="Unique identifier for the requirement")
    title: str = Field(..., description="Brief title of the requirement")
    description: str = Field(..., description="Full description of the requirement")
    section: str = Field(..., description="Section where this requirement was found")
    page_number: int | None = Field(default=None, description="Page number if available")
    priority: str | None = Field(default=None, description="Priority: high, medium, low")
    category: str | None = Field(default=None, description="Category of requirement")
    related_requirements: list[str] = Field(
        default_factory=list,
        description="IDs of related requirements",
    )

    model_config = {"frozen": False}


class RetrievedChunk(BaseModel):
    """A retrieved document chunk with relevance info."""

    chunk_id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    relevance_score: float = Field(..., ge=0.0, description="Relevance score (0-1, higher is better)")
    source_document: str = Field(..., description="Source document name")
    section: str | None = Field(default=None)
    page_numbers: list[int] = Field(default_factory=list)

    model_config = {"frozen": False}


class AnalysisResponse(BaseModel):
    """Structured response for RFP analysis."""

    # Required fields as per document
    extracted_requirements: list[ExtractedRequirement] = Field(
        default_factory=list,
        description="List of all relevant requirements found",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why these requirements apply and how they were found",
    )

    # Optional fields
    gaps_or_conflicts: list[str] = Field(
        default_factory=list,
        description="Any missing or contradictory items identified",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Confidence level (0-100)",
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Ambiguous or unclear aspects",
    )

    # Metadata
    query: str = Field(..., description="Original query")
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Chunks retrieved for this query",
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"frozen": False}


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for a single scenario."""

    retrieval_quality: int = Field(default=0, ge=0, le=25)
    reasoning_quality: int = Field(default=0, ge=0, le=30)
    completeness: int = Field(default=0, ge=0, le=20)
    clarity: int = Field(default=0, ge=0, le=15)
    structure: int = Field(default=0, ge=0, le=10)

    @property
    def total_score(self) -> int:
        """Calculate total score."""
        return (
            self.retrieval_quality
            + self.reasoning_quality
            + self.completeness
            + self.clarity
            + self.structure
        )

    @property
    def is_passing(self) -> bool:
        """Check if score meets 80+ threshold."""
        return self.total_score >= 80


class TestScenarioResult(BaseModel):
    """Result of a single test scenario."""

    scenario_name: str
    scenario_type: str
    query: str
    response: AnalysisResponse
    metrics: EvaluationMetrics
    notes: str = Field(default="")


class EvaluationResult(BaseModel):
    """Complete evaluation results."""

    scenarios: list[TestScenarioResult] = Field(default_factory=list)
    average_score: float = Field(default=0.0)
    passing_scenarios: int = Field(default=0)
    total_scenarios: int = Field(default=0)
    evaluation_timestamp: datetime = Field(default_factory=datetime.utcnow)

