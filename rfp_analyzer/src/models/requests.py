"""Request models for RFP Analyzer."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of queries the system can handle."""

    REQUIREMENT_EXTRACTION = "requirement_extraction"
    GAP_ANALYSIS = "gap_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    CONFLICT_DETECTION = "conflict_detection"
    AMBIGUITY_ANALYSIS = "ambiguity_analysis"
    GENERAL = "general"


class AnalysisRequest(BaseModel):
    """Request model for RFP analysis."""

    query: str = Field(..., min_length=5, max_length=2000, description="User's question or request")
    project_name: str | None = Field(
        default=None,
        description="Optional project name to filter documents",
    )
    query_type: QueryType = Field(
        default=QueryType.GENERAL,
        description="Type of analysis to perform",
    )
    
    # Optional parameters for advanced queries
    target_section: str | None = Field(
        default=None,
        description="Specific section to focus on",
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Additional context for the query",
    )
    additional_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context parameters for specific query types",
    )
    include_reasoning: bool = Field(
        default=True,
        description="Whether to include detailed reasoning",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What are the technical requirements for the security subsystem?",
                    "project_name": "Project Alpha",
                    "query_type": "requirement_extraction",
                },
                {
                    "query": "Does our authentication approach satisfy the RFP security requirements?",
                    "project_name": "Project Beta",
                    "query_type": "compliance_check",
                    "context": {"our_approach": "OAuth 2.0 with MFA"},
                },
            ]
        }
    }


class BatchAnalysisRequest(BaseModel):
    """Request for batch analysis of multiple queries."""

    queries: list[AnalysisRequest] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of analysis requests",
    )
    parallel: bool = Field(
        default=True,
        description="Whether to process queries in parallel",
    )

