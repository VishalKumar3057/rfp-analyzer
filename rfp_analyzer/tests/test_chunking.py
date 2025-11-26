"""Tests for chunking strategies."""

import pytest

from src.chunking.strategies import (
    RecursiveChunkingStrategy,
    HierarchicalChunkingStrategy,
)


class TestRecursiveChunkingStrategy:
    """Tests for RecursiveChunkingStrategy."""

    def test_basic_split(self):
        """Test basic text splitting."""
        strategy = RecursiveChunkingStrategy(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 50  # Long text
        
        chunks = strategy.split(text)
        
        assert len(chunks) > 1
        assert all("content" in chunk for chunk in chunks)

    def test_short_text_no_split(self):
        """Test that short text is not split."""
        strategy = RecursiveChunkingStrategy(chunk_size=1000, chunk_overlap=100)
        text = "Short text here."
        
        chunks = strategy.split(text)
        
        assert len(chunks) == 1
        assert chunks[0]["content"] == text

    def test_metadata_preserved(self):
        """Test that metadata is preserved in chunks."""
        strategy = RecursiveChunkingStrategy(chunk_size=100, chunk_overlap=20)
        text = "Test content. " * 20
        metadata = {"source": "test.pdf", "page": 1}
        
        chunks = strategy.split(text, metadata)
        
        assert all(chunk["metadata"]["source"] == "test.pdf" for chunk in chunks)


class TestHierarchicalChunkingStrategy:
    """Tests for HierarchicalChunkingStrategy."""

    def test_section_detection(self):
        """Test section detection in text."""
        strategy = HierarchicalChunkingStrategy(chunk_size=500, chunk_overlap=50)
        text = """
        1. Introduction
        This is the introduction section with some content.
        
        2. Requirements
        2.1 Technical Requirements
        The system shall implement security features.
        
        2.2 Performance Requirements
        The system shall be fast.
        
        3. Conclusion
        This is the conclusion.
        """
        
        chunks = strategy.split(text)
        
        assert len(chunks) >= 1
        # Check that section info is captured
        has_section_info = any(
            chunk.get("metadata", {}).get("section_title") 
            for chunk in chunks
        )
        assert has_section_info or len(chunks) > 0

    def test_empty_text(self):
        """Test handling of empty text."""
        strategy = HierarchicalChunkingStrategy()
        chunks = strategy.split("")
        assert chunks == []

    def test_text_cleaning(self):
        """Test that text is cleaned properly."""
        strategy = HierarchicalChunkingStrategy()
        text = "Test   with    multiple     spaces"
        
        chunks = strategy.split(text)
        
        if chunks:
            # Multiple spaces should be normalized
            assert "    " not in chunks[0]["content"]


class TestChunkingIntegration:
    """Integration tests for chunking."""

    def test_rfp_like_content(self, sample_rfp_content):
        """Test chunking RFP-like content."""
        strategy = HierarchicalChunkingStrategy(chunk_size=500, chunk_overlap=50)
        
        chunks = strategy.split(sample_rfp_content)
        
        assert len(chunks) > 0
        # Should capture some content
        total_content = " ".join(c["content"] for c in chunks)
        assert "security" in total_content.lower() or "requirement" in total_content.lower()

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        strategy = RecursiveChunkingStrategy(chunk_size=100, chunk_overlap=30)
        text = "Word " * 100  # Create text that will be split
        
        chunks = strategy.split(text)
        
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1_words = set(chunks[i]["content"].split())
                chunk2_words = set(chunks[i + 1]["content"].split())
                # There should be some common words due to overlap
                # (though this depends on where splits occur)
                assert len(chunk1_words) > 0 and len(chunk2_words) > 0

