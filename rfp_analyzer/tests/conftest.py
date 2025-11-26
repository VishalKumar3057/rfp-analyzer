"""Pytest configuration and fixtures."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Set test environment variables before importing modules
os.environ["OPENAI_API_KEY"] = "test-api-key"
os.environ["LANGCHAIN_API_KEY"] = "test-langchain-key"
os.environ["ENVIRONMENT"] = "test"


@pytest.fixture
def sample_rfp_content() -> str:
    """Sample RFP document content for testing."""
    return """
    REQUEST FOR PROPOSAL
    Project: Enterprise Security System
    
    Section 1: Technical Requirements
    
    1.1 Security Requirements
    REQ-001: The system shall implement multi-factor authentication.
    REQ-002: All data must be encrypted at rest using AES-256.
    REQ-003: The system shall maintain audit logs for all user actions.
    
    1.2 Performance Requirements
    REQ-004: The system shall support 10,000 concurrent users.
    REQ-005: Response time shall not exceed 200ms for 95% of requests.
    
    Section 2: Timeline and Budget
    
    2.1 Project Timeline
    - Phase 1: Requirements gathering (2 months)
    - Phase 2: Development (4 months)
    - Phase 3: Testing (2 months)
    - Phase 4: Deployment (1 month)
    
    2.2 Budget Constraints
    Total budget: $500,000
    - Development: $300,000
    - Infrastructure: $100,000
    - Testing: $50,000
    - Contingency: $50,000
    
    Section 3: Compliance
    
    3.1 Regulatory Requirements
    The system must comply with:
    - SOC 2 Type II
    - GDPR
    - HIPAA (if handling health data)
    """


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Sample document chunks for testing."""
    return [
        {
            "content": "REQ-001: The system shall implement multi-factor authentication.",
            "metadata": {
                "section_title": "Security Requirements",
                "section_number": "1.1",
                "page_numbers": [1],
                "contains_requirements": True,
            },
        },
        {
            "content": "REQ-002: All data must be encrypted at rest using AES-256.",
            "metadata": {
                "section_title": "Security Requirements",
                "section_number": "1.1",
                "page_numbers": [1],
                "contains_requirements": True,
            },
        },
        {
            "content": "The system shall support 10,000 concurrent users.",
            "metadata": {
                "section_title": "Performance Requirements",
                "section_number": "1.2",
                "page_numbers": [2],
                "contains_requirements": True,
            },
        },
    ]


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "extracted_requirements": [
            {
                "requirement_id": "REQ-001",
                "title": "Multi-factor Authentication",
                "description": "The system shall implement multi-factor authentication.",
                "section": "1.1 Security Requirements",
                "priority": "high",
                "category": "security",
            }
        ],
        "reasoning": "Found security requirement for MFA in section 1.1",
        "gaps_or_conflicts": [],
        "confidence": 85,
        "uncertainties": [],
    }


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock = MagicMock()
    mock.similarity_search.return_value = []
    mock.add_chunks.return_value = ["chunk-1", "chunk-2"]
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content='{"extracted_requirements": [], "reasoning": "Test", "confidence": 80}'
    )
    return mock


@pytest.fixture
def data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent.parent / "data"

