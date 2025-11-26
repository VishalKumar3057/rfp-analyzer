"""Evaluation metrics for RFP analysis."""

from typing import Any

from pydantic import BaseModel, Field

from src.models.responses import AnalysisResponse


class EvaluationMetrics(BaseModel):
    """Metrics for evaluating analysis quality."""

    # Core metrics
    requirements_found: int = Field(default=0, description="Number of requirements extracted")
    requirements_with_ids: int = Field(default=0, description="Requirements with proper IDs")
    requirements_with_sections: int = Field(default=0, description="Requirements with section refs")
    
    # Quality metrics
    reasoning_length: int = Field(default=0, description="Length of reasoning text")
    confidence_score: float = Field(default=0.0, description="Model confidence")
    gaps_identified: int = Field(default=0, description="Number of gaps identified")
    uncertainties_noted: int = Field(default=0, description="Number of uncertainties noted")
    
    # Retrieval metrics
    chunks_retrieved: int = Field(default=0, description="Number of chunks retrieved")
    avg_relevance_score: float = Field(default=0.0, description="Average relevance score")
    
    # Format metrics
    valid_json: bool = Field(default=True, description="Response is valid JSON")
    has_required_fields: bool = Field(default=True, description="Has all required fields")

    @classmethod
    def from_response(cls, response: AnalysisResponse) -> "EvaluationMetrics":
        """Create metrics from an analysis response.

        Args:
            response: Analysis response to evaluate.

        Returns:
            Computed metrics.
        """
        requirements = response.extracted_requirements
        chunks = response.retrieved_chunks or []

        # Calculate average relevance
        avg_relevance = 0.0
        if chunks:
            avg_relevance = sum(c.relevance_score for c in chunks) / len(chunks)

        return cls(
            requirements_found=len(requirements),
            requirements_with_ids=sum(1 for r in requirements if r.requirement_id),
            requirements_with_sections=sum(1 for r in requirements if r.section),
            reasoning_length=len(response.reasoning),
            confidence_score=response.confidence,
            gaps_identified=len(response.gaps_or_conflicts or []),
            uncertainties_noted=len(response.uncertainties or []),
            chunks_retrieved=len(chunks),
            avg_relevance_score=avg_relevance,
            valid_json=True,  # If we got here, JSON was valid
            has_required_fields=bool(response.reasoning),
        )


class ScenarioScore(BaseModel):
    """Score for a single test scenario."""

    scenario_name: str
    total_score: float = Field(ge=0, le=100)
    component_scores: dict[str, float] = Field(default_factory=dict)
    passed: bool = Field(default=False)
    feedback: list[str] = Field(default_factory=list)
    metrics: EvaluationMetrics | None = None


class EvaluationResult(BaseModel):
    """Overall evaluation result."""

    total_score: float = Field(ge=0, le=100)
    scenario_scores: list[ScenarioScore] = Field(default_factory=list)
    passed_scenarios: int = Field(default=0)
    failed_scenarios: int = Field(default=0)
    overall_feedback: str = ""
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        total = self.passed_scenarios + self.failed_scenarios
        return (self.passed_scenarios / total * 100) if total > 0 else 0.0


def calculate_requirement_score(
    response: AnalysisResponse,
    expected: dict[str, Any],
    criteria: dict[str, float],
) -> tuple[float, list[str]]:
    """Calculate score for requirement extraction.

    Args:
        response: Analysis response.
        expected: Expected outputs.
        criteria: Scoring criteria.

    Returns:
        Tuple of (score, feedback list).
    """
    score = 0.0
    feedback = []
    requirements = response.extracted_requirements

    # Requirements found
    if requirements:
        min_req = expected.get("min_requirements", 1)
        if len(requirements) >= min_req:
            score += criteria.get("requirements_found", 0)
            feedback.append(f"✓ Found {len(requirements)} requirements (min: {min_req})")
        else:
            partial = (len(requirements) / min_req) * criteria.get("requirements_found", 0)
            score += partial
            feedback.append(f"△ Found {len(requirements)} requirements, expected at least {min_req}")
    else:
        feedback.append("✗ No requirements found")

    # Requirements quality
    quality_score = 0.0
    for req in requirements:
        if req.requirement_id:
            quality_score += 0.25
        if req.description and len(req.description) > 20:
            quality_score += 0.25
        if req.section:
            quality_score += 0.25
        if req.priority:
            quality_score += 0.25
    
    if requirements:
        quality_score = (quality_score / len(requirements)) * criteria.get("requirements_quality", 0)
        score += quality_score
        feedback.append(f"✓ Requirements quality score: {quality_score:.1f}")

    # Section references
    with_sections = sum(1 for r in requirements if r.section)
    if with_sections > 0:
        section_score = (with_sections / max(len(requirements), 1)) * criteria.get("section_references", 0)
        score += section_score
        feedback.append(f"✓ {with_sections} requirements have section references")

    # Reasoning quality
    if response.reasoning and len(response.reasoning) > 50:
        score += criteria.get("reasoning_quality", 0)
        feedback.append("✓ Detailed reasoning provided")
    elif response.reasoning:
        score += criteria.get("reasoning_quality", 0) * 0.5
        feedback.append("△ Reasoning provided but brief")

    # Format correctness
    score += criteria.get("format_correctness", 0)
    feedback.append("✓ Response format is correct")

    return score, feedback
