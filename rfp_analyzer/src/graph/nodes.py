"""Graph nodes for RFP analysis workflow."""

from typing import Any

from langsmith import traceable

from src.graph.state import GraphState
from src.llm.analyzer import RFPAnalyzer
from src.models.requests import AnalysisRequest, QueryType
from src.models.responses import RetrievedChunk
from src.retrieval.pipeline import RetrievalPipeline
from src.utils.logging import LoggerMixin


class GraphNodes(LoggerMixin):
    """Collection of nodes for the RFP analysis graph."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        analyzer: RFPAnalyzer,
    ):
        """Initialize graph nodes.

        Args:
            retrieval_pipeline: Retrieval pipeline instance.
            analyzer: RFP analyzer instance.
        """
        self._retrieval = retrieval_pipeline
        self._analyzer = analyzer

    @traceable(name="process_query")
    def process_query(self, state: GraphState) -> dict[str, Any]:
        """Process and understand the user query.

        Args:
            state: Current graph state.

        Returns:
            Updated state fields.
        """
        self.log_info("Processing query", query=state.query[:50])

        try:
            # Detect query type if not specified
            if state.request and state.request.query_type == QueryType.GENERAL:
                detected_type = self._analyzer.detect_query_type(state.query)
            else:
                detected_type = state.request.query_type if state.request else QueryType.GENERAL

            # Extract project filter from query if not specified
            project_filter = state.project_filter
            if not project_filter and state.request:
                project_filter = state.request.project_name

            return {
                "query_type": detected_type,
                "project_filter": project_filter,
                "current_step": "query_processed",
            }

        except Exception as e:
            self.log_error("Query processing failed", error=str(e))
            return {"error": str(e), "current_step": "error"}

    @traceable(name="retrieve_documents")
    def retrieve_documents(self, state: GraphState) -> dict[str, Any]:
        """Retrieve relevant documents.

        Args:
            state: Current graph state.

        Returns:
            Updated state with retrieved chunks.
        """
        self.log_info("Retrieving documents", query_type=state.query_type.value)

        try:
            # Determine section filter based on query type
            section_filter = None
            if state.request and state.request.target_section:
                section_filter = state.request.target_section

            # Retrieve chunks
            chunks = self._retrieval.retrieve(
                query=state.query,
                project_name=state.project_filter,
                section_filter=section_filter,
                include_enrichment=True,
            )

            # Build context string
            context = self._build_context(chunks)

            return {
                "retrieved_chunks": chunks,
                "context": context,
                "current_step": "documents_retrieved",
            }

        except Exception as e:
            self.log_error("Document retrieval failed", error=str(e))
            return {"error": str(e), "current_step": "error"}

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Build context string from chunks."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            section = f"[{chunk.section}]" if chunk.section else ""
            parts.append(f"--- Chunk {i} {section} ---\n{chunk.content}\n")
        return "\n".join(parts)

    @traceable(name="analyze_content")
    def analyze_content(self, state: GraphState) -> dict[str, Any]:
        """Analyze retrieved content with LLM.

        Args:
            state: Current graph state.

        Returns:
            Updated state with analysis response.
        """
        self.log_info("Analyzing content", chunks_count=len(state.retrieved_chunks))

        try:
            if not state.request:
                # Create a default request
                request = AnalysisRequest(
                    query=state.query,
                    query_type=state.query_type,
                    project_name=state.project_filter,
                )
            else:
                request = state.request

            # Perform analysis
            response = self._analyzer.analyze(
                request=request,
                retrieved_chunks=state.retrieved_chunks,
            )

            return {
                "analysis_response": response,
                "current_step": "analysis_complete",
            }

        except Exception as e:
            self.log_error("Analysis failed", error=str(e))
            return {"error": str(e), "current_step": "error"}

    @traceable(name="handle_error")
    def handle_error(self, state: GraphState) -> dict[str, Any]:
        """Handle errors in the workflow.

        Args:
            state: Current graph state.

        Returns:
            Updated state with error handling.
        """
        self.log_warning("Handling error", error=state.error)

        from src.models.responses import AnalysisResponse

        # Create error response
        error_response = AnalysisResponse(
            extracted_requirements=[],
            reasoning=f"An error occurred during analysis: {state.error}",
            confidence=0,
            uncertainties=["Analysis could not be completed due to an error"],
            query=state.query or "Unknown query",
        )

        return {
            "analysis_response": error_response,
            "current_step": "error_handled",
        }
