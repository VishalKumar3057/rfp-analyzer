"""Main entry point for RFP Analyzer."""

import argparse
from pathlib import Path

from src.chunking.intelligent_chunker import IntelligentChunker
from src.config import get_settings
from src.evaluation.evaluator import RFPEvaluator
from src.graph.workflow import RFPAnalysisGraph
from src.loaders.pdf_loader import PDFLoader
from src.models.requests import AnalysisRequest
from src.vectorstore.chroma_store import ChromaVectorStore
from src.utils.logging import get_logger


logger = get_logger(__name__)


def ingest_documents(data_dir: Path, project_name: str = "default") -> None:
    """Ingest all PDF documents from a directory.

    Args:
        data_dir: Directory containing PDF files.
        project_name: Project name for filtering.
    """
    logger.info(f"Ingesting documents from {data_dir}")

    chunker = IntelligentChunker()
    vector_store = ChromaVectorStore()

    pdf_files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.PDF"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    total_chunks = 0
    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing: {pdf_path.name}")

            # Extract project name from filename if not specified
            doc_project = project_name
            if project_name == "default":
                doc_project = pdf_path.stem

            # Load document - create loader per file
            loader = PDFLoader(file_path=pdf_path, project_name=doc_project)
            document = loader.load()

            # Chunk document
            chunks = chunker.chunk_document(document)

            # Add to vector store
            chunk_ids = vector_store.add_chunks(chunks)
            total_chunks += len(chunk_ids)

            logger.info(f"Added {len(chunk_ids)} chunks from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    logger.info(f"Ingestion complete. Total chunks: {total_chunks}")


def run_query(query: str, project_name: str | None = None) -> None:
    """Run a single query against the RFP documents.

    Args:
        query: User query.
        project_name: Optional project filter.
    """
    logger.info(f"Running query: {query[:50]}...")

    graph = RFPAnalysisGraph()
    request = AnalysisRequest(query=query, project_name=project_name)
    
    response = graph.run(request)

    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nConfidence: {response.confidence}%")
    print(f"\nReasoning:\n{response.reasoning}")
    
    if response.extracted_requirements:
        print(f"\nExtracted Requirements ({len(response.extracted_requirements)}):")
        for req in response.extracted_requirements:
            print(f"  - [{req.requirement_id}] {req.title}")
            if req.description:
                print(f"    {req.description[:100]}...")
    
    if response.gaps_or_conflicts:
        print(f"\nGaps/Conflicts ({len(response.gaps_or_conflicts)}):")
        for gap in response.gaps_or_conflicts:
            print(f"  - {gap}")

    print("\n" + "=" * 60)


def run_evaluation() -> None:
    """Run the full evaluation suite."""
    import json
    import csv
    from datetime import datetime
    from pathlib import Path

    logger.info("Running evaluation suite")

    graph = RFPAnalysisGraph()
    evaluator = RFPEvaluator(graph)

    result = evaluator.evaluate_all()

    # Save results to JSON
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create JSON result
    json_result = {
        "timestamp": timestamp,
        "overall_score": result.total_score,
        "passed_scenarios": result.passed_scenarios,
        "failed_scenarios": result.failed_scenarios,
        "pass_rate": result.pass_rate,
        "feedback": result.overall_feedback,
        "scenarios": []
    }

    for score in result.scenario_scores:
        scenario_data = {
            "name": score.scenario_name,
            "score": score.total_score,
            "passed": score.passed,
            "feedback": score.feedback,
            "component_scores": score.component_scores or {}
        }
        json_result["scenarios"].append(scenario_data)

    # Save JSON
    json_path = results_dir / f"evaluation_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)

    # Save CSV
    csv_path = results_dir / f"evaluation_results_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Scenario", "Score", "Passed", "Feedback"])
        for score in result.scenario_scores:
            # Clean feedback of special characters for CSV
            clean_feedback = "; ".join(score.feedback[:3]).replace("✓", "[OK]").replace("△", "[WARN]").replace("✗", "[FAIL]")
            writer.writerow([
                score.scenario_name,
                score.total_score,
                "PASS" if score.passed else "FAIL",
                clean_feedback
            ])
        writer.writerow([])
        writer.writerow(["OVERALL", result.total_score, f"{result.passed_scenarios}/5 passed", result.overall_feedback])

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Score: {result.total_score:.1f}/100")
    print(f"Passed: {result.passed_scenarios}/5")
    print(f"Pass Rate: {result.pass_rate:.1f}%")
    print(f"\n{result.overall_feedback}")

    print("\nScenario Details:")
    for score in result.scenario_scores:
        status = "✓ PASS" if score.passed else "✗ FAIL"
        print(f"\n  {score.scenario_name}: {score.total_score:.1f}/100 [{status}]")
        for fb in score.feedback[:3]:
            print(f"    {fb}")

    print("\n" + "=" * 60)
    print(f"\nResults saved to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RFP Analyzer - AI-powered RFP analysis")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF documents")
    ingest_parser.add_argument("--data-dir", type=Path, required=True, help="Directory with PDFs")
    ingest_parser.add_argument("--project", default="default", help="Project name")

    # Query command
    query_parser = subparsers.add_parser("query", help="Run a query")
    query_parser.add_argument("query", type=str, help="Query text")
    query_parser.add_argument("--project", default=None, help="Project filter")

    # Evaluate command
    subparsers.add_parser("evaluate", help="Run evaluation suite")

    # Server command
    server_parser = subparsers.add_parser("serve", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host")
    server_parser.add_argument("--port", type=int, default=8000, help="Port")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_documents(args.data_dir, args.project)
    elif args.command == "query":
        run_query(args.query, args.project)
    elif args.command == "evaluate":
        run_evaluation()
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

