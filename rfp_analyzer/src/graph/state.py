"""Graph state definitions for LangGraph workflow."""

from typing import Annotated, Any
from operator import add

from pydantic import BaseModel, Field
from langgraph.graph import add_messages

from src.models.requests import AnalysisRequest, QueryType
from src.models.responses import AnalysisResponse, RetrievedChunk


class GraphState(BaseModel):
    """State object for the RFP analysis graph.
    
    This state is passed between nodes in the LangGraph workflow.
    """

    # Input
    request: AnalysisRequest | None = Field(
        default=None,
        description="The original analysis request",
    )
    
    # Query processing
    query: str = Field(
        default="",
        description="The processed query",
    )
    query_type: QueryType = Field(
        default=QueryType.GENERAL,
        description="Detected or specified query type",
    )
    project_filter: str | None = Field(
        default=None,
        description="Project name for filtering",
    )
    
    # Retrieval
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Retrieved document chunks",
    )
    context: str = Field(
        default="",
        description="Formatted context for LLM",
    )
    
    # Analysis
    analysis_response: AnalysisResponse | None = Field(
        default=None,
        description="The final analysis response",
    )
    
    # Workflow control
    error: str | None = Field(
        default=None,
        description="Error message if any",
    )
    current_step: str = Field(
        default="start",
        description="Current step in the workflow",
    )
    
    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    model_config = {"arbitrary_types_allowed": True}


class ConversationState(BaseModel):
    """State for multi-turn conversations."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="Conversation messages",
    )
    graph_state: GraphState = Field(
        default_factory=GraphState,
        description="Current graph state",
    )
    history: list[AnalysisResponse] = Field(
        default_factory=list,
        description="Previous analysis responses",
    )

    model_config = {"arbitrary_types_allowed": True}

