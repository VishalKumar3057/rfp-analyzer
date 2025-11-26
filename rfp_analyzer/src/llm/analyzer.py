"""RFP Analyzer using LLM for analysis."""

from typing import Any

from langchain_openai import ChatOpenAI
from langsmith import traceable

from src.config import get_settings
from src.llm.output_parser import StructuredOutputParser
from src.llm.prompts import PromptTemplates
from src.models.requests import AnalysisRequest, QueryType
from src.models.responses import AnalysisResponse, RetrievedChunk
from src.utils.logging import LoggerMixin


class RFPAnalyzer(LoggerMixin):
    """LLM-based RFP analyzer with structured output."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.1,
    ):
        """Initialize RFP analyzer.

        Args:
            model: OpenAI model name.
            temperature: LLM temperature.
        """
        settings = get_settings()
        
        self._llm = ChatOpenAI(
            model=model or settings.openai_model,
            temperature=temperature,
            openai_api_key=settings.openai_api_key.get_secret_value(),
        )
        self._parser = StructuredOutputParser()
        
        self.log_info(
            "RFP Analyzer initialized",
            model=model or settings.openai_model,
        )

    @traceable(name="analyze_rfp")
    def analyze(
        self,
        request: AnalysisRequest,
        retrieved_chunks: list[RetrievedChunk],
    ) -> AnalysisResponse:
        """Analyze RFP based on request and retrieved context.

        Args:
            request: Analysis request with query and parameters.
            retrieved_chunks: Retrieved document chunks.

        Returns:
            Structured analysis response.
        """
        self.log_info(
            "Starting analysis",
            query_type=request.query_type.value,
            chunks_count=len(retrieved_chunks),
        )

        # Build context from chunks
        context = self._build_context(retrieved_chunks)

        # Get appropriate prompt template
        template_name = self._get_template_name(request.query_type)
        prompt = PromptTemplates.get_chat_prompt(template_name)

        # Prepare prompt variables
        prompt_vars = self._prepare_prompt_vars(request, context)

        # Generate response
        chain = prompt | self._llm
        response = chain.invoke(prompt_vars)

        # Parse structured output
        analysis = self._parser.parse(response.content, query=request.query)

        # Add retrieved chunks to response
        analysis.retrieved_chunks = retrieved_chunks

        self.log_info(
            "Analysis complete",
            requirements_found=len(analysis.extracted_requirements),
            confidence=analysis.confidence,
        )

        return analysis

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            section_info = f"[Section: {chunk.section}]" if chunk.section else ""
            page_info = f"[Pages: {chunk.page_numbers}]" if chunk.page_numbers else ""
            
            context_parts.append(
                f"--- Document Chunk {i} {section_info} {page_info} ---\n"
                f"{chunk.content}\n"
            )
        
        return "\n".join(context_parts)

    def _get_template_name(self, query_type: QueryType) -> str:
        """Map query type to template name."""
        mapping = {
            QueryType.REQUIREMENT_EXTRACTION: "requirement_extraction",
            QueryType.GAP_ANALYSIS: "gap_analysis",
            QueryType.COMPLIANCE_CHECK: "compliance_check",
            QueryType.CONFLICT_DETECTION: "conflict_detection",
            QueryType.AMBIGUITY_ANALYSIS: "ambiguity_analysis",
            QueryType.GENERAL: "general",
        }
        return mapping.get(query_type, "general")

    def _prepare_prompt_vars(
        self,
        request: AnalysisRequest,
        context: str,
    ) -> dict[str, Any]:
        """Prepare variables for prompt template."""
        base_vars = {
            "query": request.query,
            "context": context,
        }

        # Add type-specific variables
        if request.query_type == QueryType.COMPLIANCE_CHECK:
            base_vars["approach"] = request.additional_context.get("approach", "")
            base_vars["section"] = request.additional_context.get("section", "")
        
        elif request.query_type == QueryType.GAP_ANALYSIS:
            base_vars["approach"] = request.additional_context.get("approach", "")
        
        elif request.query_type == QueryType.CONFLICT_DETECTION:
            base_vars["timeline"] = request.additional_context.get("timeline", "")
            base_vars["budget"] = request.additional_context.get("budget", "")
            base_vars["scope"] = request.additional_context.get("scope", "")
        
        elif request.query_type == QueryType.AMBIGUITY_ANALYSIS:
            base_vars["term"] = request.additional_context.get("term", request.query)

        return base_vars

    @traceable(name="quick_analyze")
    def quick_analyze(
        self,
        query: str,
        context: str,
    ) -> AnalysisResponse:
        """Quick analysis without full request object.

        Args:
            query: User query.
            context: Document context.

        Returns:
            Analysis response.
        """
        prompt = PromptTemplates.get_chat_prompt("general")

        chain = prompt | self._llm
        response = chain.invoke({"query": query, "context": context})

        return self._parser.parse(response.content, query=query)

    def detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query from the text.

        Args:
            query: User query.

        Returns:
            Detected query type.
        """
        query_lower = query.lower()

        # Requirement extraction patterns
        if any(
            term in query_lower
            for term in ["requirements for", "what are the requirements", "list requirements"]
        ):
            return QueryType.REQUIREMENT_EXTRACTION

        # Gap analysis patterns
        if any(
            term in query_lower
            for term in ["missing", "gaps", "what might we be missing"]
        ):
            return QueryType.GAP_ANALYSIS

        # Compliance patterns
        if any(
            term in query_lower
            for term in ["compliant", "compliance", "does our approach", "is our"]
        ):
            return QueryType.COMPLIANCE_CHECK

        # Conflict patterns
        if any(
            term in query_lower
            for term in ["conflict", "realistic", "timeline", "budget", "scope together"]
        ):
            return QueryType.CONFLICT_DETECTION

        # Ambiguity patterns
        if any(
            term in query_lower
            for term in ["ambiguous", "unclear", "what does", "meaning of"]
        ):
            return QueryType.AMBIGUITY_ANALYSIS

        return QueryType.GENERAL
