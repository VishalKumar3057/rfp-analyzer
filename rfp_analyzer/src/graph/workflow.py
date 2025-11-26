"""LangGraph workflow for RFP analysis."""

from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langsmith import traceable

from src.config import get_settings
from src.graph.nodes import GraphNodes
from src.graph.state import GraphState
from src.llm.analyzer import RFPAnalyzer
from src.models.requests import AnalysisRequest
from src.models.responses import AnalysisResponse
from src.retrieval.pipeline import RetrievalPipeline
from src.vectorstore.chroma_store import ChromaVectorStore
from src.utils.logging import LoggerMixin


class RFPAnalysisGraph(LoggerMixin):
    """LangGraph-based RFP analysis workflow."""

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
        analyzer: RFPAnalyzer | None = None,
    ):
        """Initialize the RFP analysis graph.

        Args:
            vector_store: Optional vector store instance.
            retrieval_pipeline: Optional retrieval pipeline.
            analyzer: Optional RFP analyzer.
        """
        # Initialize components
        self._vector_store = vector_store or ChromaVectorStore()
        self._retrieval = retrieval_pipeline or RetrievalPipeline(self._vector_store)
        self._analyzer = analyzer or RFPAnalyzer()
        
        # Initialize nodes
        self._nodes = GraphNodes(
            retrieval_pipeline=self._retrieval,
            analyzer=self._analyzer,
        )
        
        # Build the graph
        self._graph = self._build_graph()
        
        self.log_info("RFP Analysis Graph initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create graph with state schema
        graph = StateGraph(GraphState)

        # Add nodes
        graph.add_node("process_query", self._nodes.process_query)
        graph.add_node("retrieve_documents", self._nodes.retrieve_documents)
        graph.add_node("analyze_content", self._nodes.analyze_content)
        graph.add_node("handle_error", self._nodes.handle_error)

        # Set entry point
        graph.set_entry_point("process_query")

        # Add conditional edges
        graph.add_conditional_edges(
            "process_query",
            self._route_after_query,
            {
                "retrieve": "retrieve_documents",
                "error": "handle_error",
            },
        )

        graph.add_conditional_edges(
            "retrieve_documents",
            self._route_after_retrieval,
            {
                "analyze": "analyze_content",
                "error": "handle_error",
            },
        )

        graph.add_conditional_edges(
            "analyze_content",
            self._route_after_analysis,
            {
                "end": END,
                "error": "handle_error",
            },
        )

        graph.add_edge("handle_error", END)

        return graph.compile()

    def _route_after_query(self, state: GraphState) -> Literal["retrieve", "error"]:
        """Route after query processing."""
        if state.error:
            return "error"
        return "retrieve"

    def _route_after_retrieval(self, state: GraphState) -> Literal["analyze", "error"]:
        """Route after document retrieval."""
        if state.error:
            return "error"
        if not state.retrieved_chunks:
            # No documents found, but continue to analysis for proper response
            return "analyze"
        return "analyze"

    def _route_after_analysis(self, state: GraphState) -> Literal["end", "error"]:
        """Route after content analysis."""
        if state.error:
            return "error"
        return "end"

    @traceable(name="run_analysis")
    def run(self, request: AnalysisRequest) -> AnalysisResponse:
        """Run the analysis workflow.

        Args:
            request: Analysis request.

        Returns:
            Analysis response.
        """
        self.log_info("Starting analysis workflow", query=request.query[:50])

        # Create initial state
        initial_state = GraphState(
            request=request,
            query=request.query,
            project_filter=request.project_name,
            query_type=request.query_type,
        )

        # Run the graph
        final_state = self._graph.invoke(initial_state)

        # Extract response
        if isinstance(final_state, dict):
            response = final_state.get("analysis_response")
        else:
            response = final_state.analysis_response

        if not response:
            response = AnalysisResponse(
                extracted_requirements=[],
                reasoning="No response generated",
                confidence=0,
            )

        self.log_info(
            "Analysis workflow complete",
            requirements_count=len(response.extracted_requirements),
        )

        return response

    @traceable(name="quick_query")
    def quick_query(self, query: str, project_name: str | None = None) -> AnalysisResponse:
        """Quick query without full request object.

        Args:
            query: User query.
            project_name: Optional project filter.

        Returns:
            Analysis response.
        """
        request = AnalysisRequest(
            query=query,
            project_name=project_name,
        )
        return self.run(request)
