"""API dependencies for dependency injection."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.chunking.intelligent_chunker import IntelligentChunker
from src.graph.workflow import RFPAnalysisGraph
from src.llm.analyzer import RFPAnalyzer
from src.loaders.pdf_loader import PDFLoader
from src.retrieval.pipeline import RetrievalPipeline
from src.vectorstore.chroma_store import ChromaVectorStore


@lru_cache()
def get_vector_store() -> ChromaVectorStore:
    """Get or create vector store instance."""
    return ChromaVectorStore()


@lru_cache()
def get_retrieval_pipeline() -> RetrievalPipeline:
    """Get or create retrieval pipeline instance."""
    vector_store = get_vector_store()
    return RetrievalPipeline(vector_store=vector_store)


@lru_cache()
def get_analyzer() -> RFPAnalyzer:
    """Get or create RFP analyzer instance."""
    return RFPAnalyzer()


@lru_cache()
def get_analysis_graph() -> RFPAnalysisGraph:
    """Get or create analysis graph instance."""
    vector_store = get_vector_store()
    retrieval = get_retrieval_pipeline()
    analyzer = get_analyzer()
    return RFPAnalysisGraph(
        vector_store=vector_store,
        retrieval_pipeline=retrieval,
        analyzer=analyzer,
    )


def get_pdf_loader() -> PDFLoader:
    """Get PDF loader instance."""
    return PDFLoader()


def get_chunker() -> IntelligentChunker:
    """Get intelligent chunker instance."""
    return IntelligentChunker()


# Type aliases for dependency injection
VectorStoreDep = Annotated[ChromaVectorStore, Depends(get_vector_store)]
RetrievalDep = Annotated[RetrievalPipeline, Depends(get_retrieval_pipeline)]
AnalyzerDep = Annotated[RFPAnalyzer, Depends(get_analyzer)]
GraphDep = Annotated[RFPAnalysisGraph, Depends(get_analysis_graph)]
LoaderDep = Annotated[PDFLoader, Depends(get_pdf_loader)]
ChunkerDep = Annotated[IntelligentChunker, Depends(get_chunker)]

