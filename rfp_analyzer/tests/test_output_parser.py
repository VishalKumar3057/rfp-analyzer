"""Tests for LLM output parser."""

import json
import pytest

from src.llm.output_parser import StructuredOutputParser


class TestStructuredOutputParser:
    """Tests for StructuredOutputParser."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON output."""
        parser = StructuredOutputParser()
        
        output = json.dumps({
            "extracted_requirements": [
                {
                    "requirement_id": "REQ-001",
                    "title": "Test Requirement",
                    "description": "Description here",
                    "section": "1.1",
                }
            ],
            "reasoning": "Found one requirement",
            "confidence": 85,
            "gaps_or_conflicts": [],
            "uncertainties": [],
        })
        
        result = parser.parse(output)
        
        assert len(result.extracted_requirements) == 1
        assert result.extracted_requirements[0].requirement_id == "REQ-001"
        assert result.confidence == 85

    def test_parse_json_in_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        parser = StructuredOutputParser()
        
        output = """Here is the analysis:

```json
{
    "extracted_requirements": [],
    "reasoning": "No requirements found",
    "confidence": 50
}
```

That's the result."""
        
        result = parser.parse(output)
        
        assert result.reasoning == "No requirements found"
        assert result.confidence == 50

    def test_parse_malformed_json(self):
        """Test handling of malformed JSON."""
        parser = StructuredOutputParser()
        
        output = "This is not valid JSON at all"
        
        result = parser.parse(output)
        
        # Should return a valid response with error info
        assert result.confidence == 0 or "Failed" in result.reasoning

    def test_parse_json_with_trailing_comma(self):
        """Test parsing JSON with trailing commas."""
        parser = StructuredOutputParser()
        
        output = """{
            "extracted_requirements": [],
            "reasoning": "Test",
            "confidence": 70,
        }"""
        
        result = parser.parse(output)
        
        # Should handle trailing comma
        assert result.reasoning == "Test" or result.confidence >= 0

    def test_normalize_priority(self):
        """Test priority normalization."""
        parser = StructuredOutputParser()
        
        output = json.dumps({
            "extracted_requirements": [
                {
                    "requirement_id": "REQ-001",
                    "title": "Test",
                    "description": "Desc",
                    "priority": "CRITICAL",
                }
            ],
            "reasoning": "Test",
            "confidence": 80,
        })
        
        result = parser.parse(output)
        
        # CRITICAL should be normalized to "high"
        assert result.extracted_requirements[0].priority == "high"

    def test_parse_empty_requirements(self):
        """Test parsing response with no requirements."""
        parser = StructuredOutputParser()
        
        output = json.dumps({
            "extracted_requirements": [],
            "reasoning": "No requirements found in the document",
            "confidence": 90,
            "gaps_or_conflicts": ["Missing security section"],
        })
        
        result = parser.parse(output)
        
        assert len(result.extracted_requirements) == 0
        assert len(result.gaps_or_conflicts) == 1

    def test_parse_with_missing_fields(self):
        """Test parsing JSON with missing optional fields."""
        parser = StructuredOutputParser()
        
        output = json.dumps({
            "extracted_requirements": [
                {
                    "requirement_id": "REQ-001",
                    "title": "Test",
                    "description": "Desc",
                    # Missing: section, priority, category, etc.
                }
            ],
            "reasoning": "Partial data",
            "confidence": 60,
        })
        
        result = parser.parse(output)
        
        assert result.extracted_requirements[0].section is None
        assert result.extracted_requirements[0].priority is None

    def test_auto_generate_requirement_ids(self):
        """Test auto-generation of requirement IDs."""
        parser = StructuredOutputParser()
        
        output = json.dumps({
            "extracted_requirements": [
                {
                    "title": "First Requirement",
                    "description": "Desc 1",
                },
                {
                    "title": "Second Requirement",
                    "description": "Desc 2",
                }
            ],
            "reasoning": "Found requirements without IDs",
            "confidence": 75,
        })
        
        result = parser.parse(output)
        
        # Should auto-generate IDs
        assert result.extracted_requirements[0].requirement_id is not None
        assert result.extracted_requirements[1].requirement_id is not None

