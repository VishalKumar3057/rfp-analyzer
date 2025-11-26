"""API routes for RFP Analyzer."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from src.api.dependencies import (
    GraphDep,
    VectorStoreDep,
    LoaderDep,
    ChunkerDep,
)
from src.models.requests import AnalysisRequest, BatchAnalysisRequest
from src.models.responses import AnalysisResponse


router = APIRouter(prefix="/api/v1", tags=["RFP Analysis"])


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "1.0.0"


class IngestResponse(BaseModel):
    """Document ingestion response."""
    success: bool
    document_id: str
    chunks_created: int
    message: str


class StatsResponse(BaseModel):
    """Vector store statistics response."""
    collection_name: str
    document_count: int
    persist_directory: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_rfp(
    request: AnalysisRequest,
    graph: GraphDep,
) -> AnalysisResponse:
    """Analyze RFP documents based on the query.

    Args:
        request: Analysis request with query and parameters.
        graph: Injected analysis graph.

    Returns:
        Structured analysis response.
    """
    try:
        response = graph.run(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/batch", response_model=list[AnalysisResponse])
async def batch_analyze(
    request: BatchAnalysisRequest,
    graph: GraphDep,
) -> list[AnalysisResponse]:
    """Batch analyze multiple queries.

    Args:
        request: Batch analysis request.
        graph: Injected analysis graph.

    Returns:
        List of analysis responses.
    """
    responses = []
    for query_request in request.queries:
        try:
            response = graph.run(query_request)
            responses.append(response)
        except Exception as e:
            # Create error response for failed queries
            responses.append(
                AnalysisResponse(
                    extracted_requirements=[],
                    reasoning=f"Error: {str(e)}",
                    confidence=0,
                    query=query_request.query,
                )
            )
    return responses


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    project_name: str = Form(...),
    vector_store: VectorStoreDep = None,
    loader: LoaderDep = None,
    chunker: ChunkerDep = None,
) -> IngestResponse:
    """Ingest a PDF document into the vector store.

    Args:
        file: PDF file to ingest.
        project_name: Project name for filtering.
        vector_store: Injected vector store.
        loader: Injected PDF loader.
        chunker: Injected chunker.

    Returns:
        Ingestion result.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{file.filename}")
        content = await file.read()
        temp_path.write_bytes(content)

        # Load document
        document = loader.load(temp_path, project_name=project_name)

        # Chunk document
        chunks = chunker.chunk_document(document)

        # Add to vector store
        chunk_ids = vector_store.add_chunks(chunks)

        # Clean up
        temp_path.unlink()

        return IngestResponse(
            success=True,
            document_id=str(document.id),
            chunks_created=len(chunk_ids),
            message=f"Successfully ingested {file.filename}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats(vector_store: VectorStoreDep) -> StatsResponse:
    """Get vector store statistics."""
    stats = vector_store.get_collection_stats()
    return StatsResponse(**stats)
