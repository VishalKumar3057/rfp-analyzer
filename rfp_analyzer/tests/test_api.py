"""Tests for API endpoints."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.models.responses import AnalysisResponse


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "RFP Analyzer API"
        assert "docs" in data


class TestAnalyzeEndpoint:
    """Tests for analyze endpoint."""

    @patch("src.api.dependencies.get_analysis_graph")
    def test_analyze_success(self, mock_get_graph, client):
        """Test successful analysis request."""
        # Setup mock
        mock_graph = MagicMock()
        mock_graph.run.return_value = AnalysisResponse(
            extracted_requirements=[],
            reasoning="Test analysis",
            confidence=80,
        )
        mock_get_graph.return_value = mock_graph

        response = client.post(
            "/api/v1/analyze",
            json={
                "query": "What are the security requirements?",
                "project_name": "Test Project",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "reasoning" in data
        assert "extracted_requirements" in data

    def test_analyze_invalid_query(self, client):
        """Test analysis with invalid query."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "query": "Hi",  # Too short
            },
        )

        assert response.status_code == 422  # Validation error


class TestStatsEndpoint:
    """Tests for stats endpoint."""

    @patch("src.api.dependencies.get_vector_store")
    def test_get_stats(self, mock_get_store, client):
        """Test getting vector store stats."""
        mock_store = MagicMock()
        mock_store.get_collection_stats.return_value = {
            "collection_name": "test_collection",
            "document_count": 100,
            "persist_directory": "/tmp/test",
        }
        mock_get_store.return_value = mock_store

        response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = response.json()
        assert "collection_name" in data
        assert "document_count" in data


class TestBatchAnalyzeEndpoint:
    """Tests for batch analyze endpoint."""

    @patch("src.api.dependencies.get_analysis_graph")
    def test_batch_analyze(self, mock_get_graph, client):
        """Test batch analysis."""
        mock_graph = MagicMock()
        mock_graph.run.return_value = AnalysisResponse(
            extracted_requirements=[],
            reasoning="Test",
            confidence=75,
        )
        mock_get_graph.return_value = mock_graph

        response = client.post(
            "/api/v1/analyze/batch",
            json={
                "queries": [
                    {"query": "First query here"},
                    {"query": "Second query here"},
                ],
                "parallel": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

