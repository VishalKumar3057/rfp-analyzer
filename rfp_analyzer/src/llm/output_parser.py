"""Structured output parser for LLM responses."""

import json
import re
from typing import Any

from pydantic import ValidationError

from src.models.responses import AnalysisResponse, ExtractedRequirement
from src.utils.logging import LoggerMixin


class StructuredOutputParser(LoggerMixin):
    """Parser for structured JSON output from LLM."""

    def parse(self, llm_output: str, query: str = "Unknown query") -> AnalysisResponse:
        """Parse LLM output into structured AnalysisResponse.

        Args:
            llm_output: Raw LLM output string.
            query: Original query for the response.

        Returns:
            Parsed AnalysisResponse.
        """
        try:
            # Try to extract JSON from the output
            json_data = self._extract_json(llm_output)

            # Parse requirements
            requirements = self._parse_requirements(
                json_data.get("extracted_requirements", [])
            )

            # Build response
            return AnalysisResponse(
                extracted_requirements=requirements,
                reasoning=json_data.get("reasoning", ""),
                gaps_or_conflicts=json_data.get("gaps_or_conflicts", []),
                confidence=json_data.get("confidence", 50),
                uncertainties=json_data.get("uncertainties", []),
                query=query,
            )

        except Exception as e:
            self.log_error("Failed to parse LLM output", error=str(e))
            # Return a minimal valid response
            return AnalysisResponse(
                extracted_requirements=[],
                reasoning=f"Failed to parse response: {str(e)}. Raw output: {llm_output[:500]}",
                confidence=0,
                query=query,
            )

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from text that may contain other content."""
        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown code blocks
        json_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Clean the match
                    cleaned = match.strip()
                    if not cleaned.startswith("{"):
                        continue
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue

        # Last resort: try to fix common JSON issues
        return self._fix_and_parse_json(text)

    def _fix_and_parse_json(self, text: str) -> dict[str, Any]:
        """Attempt to fix common JSON formatting issues."""
        # Find the first { and last }
        start = text.find("{")
        end = text.rfind("}")
        
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in text")
        
        json_str = text[start : end + 1]
        
        # Common fixes
        fixes = [
            (r",\s*}", "}"),  # Remove trailing commas
            (r",\s*]", "]"),  # Remove trailing commas in arrays
            (r"'", '"'),  # Replace single quotes with double
            (r"(\w+):", r'"\1":'),  # Quote unquoted keys
        ]
        
        for pattern, replacement in fixes:
            json_str = re.sub(pattern, replacement, json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.log_warning("JSON fix failed", error=str(e))
            # Return minimal structure
            return {
                "extracted_requirements": [],
                "reasoning": text,
                "confidence": 30,
            }

    def _parse_requirements(
        self, requirements_data: list[Any]
    ) -> list[ExtractedRequirement]:
        """Parse requirement data into ExtractedRequirement objects.

        Handles both dict format and string format from LLM.
        """
        requirements = []

        for idx, req_data in enumerate(requirements_data):
            try:
                # Handle string format (LLM sometimes returns just strings)
                if isinstance(req_data, str):
                    requirement = ExtractedRequirement(
                        requirement_id=f"REQ-{idx + 1:03d}",
                        title=req_data[:100] if len(req_data) > 100 else req_data,
                        description=req_data,
                        section="Extracted from document",
                    )
                    requirements.append(requirement)
                    continue

                # Handle dict format
                if isinstance(req_data, dict):
                    requirement = ExtractedRequirement(
                        requirement_id=req_data.get("requirement_id", f"REQ-{idx + 1:03d}"),
                        title=req_data.get("title", req_data.get("description", "Untitled Requirement")[:100]),
                        description=req_data.get("description", str(req_data)),
                        section=req_data.get("section", "Unknown"),
                        page_number=req_data.get("page_number"),
                        priority=self._normalize_priority(req_data.get("priority")),
                        category=req_data.get("category"),
                        related_requirements=req_data.get("related_requirements", []),
                    )
                    requirements.append(requirement)
                else:
                    # Fallback for other types
                    requirement = ExtractedRequirement(
                        requirement_id=f"REQ-{idx + 1:03d}",
                        title=str(req_data)[:100],
                        description=str(req_data),
                        section="Extracted from document",
                    )
                    requirements.append(requirement)

            except ValidationError as e:
                self.log_warning(
                    "Failed to parse requirement",
                    index=idx,
                    error=str(e),
                )
                continue

        return requirements

    def _normalize_priority(self, priority: str | None) -> str | None:
        """Normalize priority values."""
        if not priority:
            return None
        
        priority_lower = priority.lower().strip()
        
        priority_map = {
            "high": "high",
            "critical": "high",
            "mandatory": "high",
            "medium": "medium",
            "moderate": "medium",
            "normal": "medium",
            "low": "low",
            "optional": "low",
            "nice-to-have": "low",
        }
        
        return priority_map.get(priority_lower, priority_lower)
