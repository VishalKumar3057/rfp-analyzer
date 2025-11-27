# RFP Analyzer - AI-Powered RFP Analysis System

An intelligent RFP (Request for Proposal) analysis system using **RAG (Retrieval-Augmented Generation) + LLM** that helps teams quickly find and reason about requirements scattered across complex RFP documents.

## Features

- **Smart Document Chunking**: Handles hierarchical structures, cross-references, and scattered requirements
- **Multi-Stage Retrieval Pipeline**: Semantic search → Filtering → Re-ranking → Context enrichment
- **Structured JSON Output**: Returns `extracted_requirements` and `reasoning` in a structured format
- **Project-based Filtering**: Filter queries by project name
- **REST API**: FastAPI-based API with Swagger documentation
- **Evaluation Suite**: 5 realistic test scenarios with scoring metrics

## Architecture

```
PDF Documents → Markdown Extraction → Section Splitting → Metadata Generation → Embeddings → ChromaDB
                                                                                      ↓
User Query → Query Processing → Semantic Search → Re-ranking → LLM Analysis → Structured Response
```

## Tech Stack

- **LangChain** - RAG framework
- **LangGraph** - Workflow orchestration
- **LangSmith** - Tracing and monitoring
- **OpenAI GPT-4o** - LLM for analysis
- **OpenAI text-embedding-3-large** - Embeddings
- **ChromaDB** - Vector storage
- **FastAPI** - REST API
- **Pydantic** - Data validation

## Setup Instructions

### 1. Prerequisites

- Python 3.11+
- OpenAI API Key

### 2. Installation

```bash
# Clone the repository
cd rfp_analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file:

```env
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional - LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=rfp-analyzer

# Settings
SIMILARITY_THRESHOLD=0.3
TOP_K_RESULTS=10
```

### 4. Ingest Documents

Place your RFP PDF files under the project folder, for example:

`rfp_analyzer/data/pdfs`

```bash
# Ingest RFP documents from the local data/pdfs directory
python run.py ingest --data-dir "data/pdfs" --project RFP1
```

### 5. Run Queries

```bash
# Run a query
python run.py query "What are the technical requirements for inspections?"

# Filter by project
python run.py query "What are the compliance requirements?" --project RFP1
```

### 6. Start API Server

```bash
python run.py serve

# Access Swagger UI at http://127.0.0.1:8000/docs
```

### 7. Run Evaluation

```bash
python run.py evaluate
```

### 8. Docker Usage

You can also run the RFP Analyzer inside a Docker container.

#### Build the image

From the `rfp_analyzer` directory:

```bash
docker build -t rfp-analyzer:latest .
```

#### Run the API with Docker

Expose port `8000` and pass your OpenAI key as an environment variable:

```bash
docker run --rm \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-openai-api-key \
  rfp-analyzer:latest

# Swagger UI will be available at http://127.0.0.1:8000/docs
```

#### Ingest documents inside the container

Assuming your RFP PDFs are in the project at `rfp_analyzer/data/pdfs` on your host, you can
mount that directory into the container and run ingestion there. Example:

```bash
docker run --rm \
  -e OPENAI_API_KEY=your-openai-api-key \
  -v /absolute/path/to/rfp_analyzer/data/pdfs:/data:ro \
  -v /absolute/path/to/rfp_analyzer/data:/app/data \
  rfp-analyzer:latest \
  python run.py ingest --data-dir "/data" --project RFP1

# After this, the ChromaDB index is stored under /app/data/chroma_db (mounted from host).
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analyze` | POST | Analyze RFP with a query |
| `/api/v1/stats` | GET | Get vector store statistics |
| `/api/v1/health` | GET | Health check |

### Example Request

```json
POST /api/v1/analyze
{
  "query": "What are the technical requirements for inspections?",
  "project_name": "RFP1"
}
```

### Example Response

```json
{
  "extracted_requirements": [
    {
      "requirement_id": "REQ-001",
      "title": "Visual Inspection Requirements",
      "description": "All inspections must be documented with reports...",
      "section": "Inspection Process"
    }
  ],
  "reasoning": "Based on the RFP documents, the technical requirements...",
  "confidence": 85.0,
  "query": "What are the technical requirements for inspections?"
}
```

## Project Structure

```
rfp_analyzer/
├── src/
│   ├── api/           # FastAPI routes
│   ├── chunking/      # Document chunking strategies
│   ├── config/        # Settings and configuration
│   ├── evaluation/    # Test scenarios and metrics
│   ├── graph/         # LangGraph workflow
│   ├── llm/           # LLM analyzer and prompts
│   ├── loaders/       # PDF document loaders
│   ├── models/        # Pydantic models
│   ├── retrieval/     # Multi-stage retrieval pipeline
│   ├── utils/         # Logging utilities
│   └── vectorstore/   # ChromaDB integration
├── tests/             # Unit tests
├── data/              # Vector store persistence
└── results/           # Evaluation results
```
