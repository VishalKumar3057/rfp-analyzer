"""Test scenarios for RFP analysis evaluation."""

from dataclasses import dataclass
from typing import Any

from src.models.requests import AnalysisRequest, QueryType


@dataclass
class TestScenario:
    """Definition of a test scenario."""

    name: str
    description: str
    query_type: QueryType
    request: AnalysisRequest
    expected_outputs: dict[str, Any]
    scoring_criteria: dict[str, float]
    max_score: int = 100


class TestScenarios:
    """Collection of 5 test scenarios for RFP analysis evaluation."""

    @staticmethod
    def scenario_1_requirement_extraction() -> TestScenario:
        """Scenario 1: Requirement Extraction.

        Query: "What are all the technical requirements for subsystem X?"
        Expected: List of requirements with IDs, descriptions, and sections.
        """
        return TestScenario(
            name="Requirement Extraction",
            description="Extract all technical requirements for a specific subsystem",
            query_type=QueryType.REQUIREMENT_EXTRACTION,
            request=AnalysisRequest(
                query="What are all the technical requirements for the security subsystem?",
                query_type=QueryType.REQUIREMENT_EXTRACTION,
            ),
            expected_outputs={
                "has_requirements": True,
                "min_requirements": 3,
                "has_sections": True,
                "has_reasoning": True,
            },
            scoring_criteria={
                "requirements_found": 30,
                "requirements_quality": 25,
                "section_references": 15,
                "reasoning_quality": 20,
                "format_correctness": 10,
            },
        )

    @staticmethod
    def scenario_2_gap_analysis() -> TestScenario:
        """Scenario 2: Gap Analysis.

        Query: "What requirements might we be missing?"
        Expected: Identified gaps with explanations.
        """
        return TestScenario(
            name="Gap Analysis",
            description="Identify missing or incomplete requirements",
            query_type=QueryType.GAP_ANALYSIS,
            request=AnalysisRequest(
                query="What requirements might we be missing in our security implementation?",
                query_type=QueryType.GAP_ANALYSIS,
                additional_context={
                    "approach": "We plan to implement OAuth 2.0 authentication with basic encryption.",
                },
            ),
            expected_outputs={
                "has_gaps": True,
                "has_reasoning": True,
                "actionable_insights": True,
            },
            scoring_criteria={
                "gaps_identified": 30,
                "gap_relevance": 25,
                "actionable_suggestions": 20,
                "reasoning_quality": 15,
                "format_correctness": 10,
            },
        )

    @staticmethod
    def scenario_3_compliance_check() -> TestScenario:
        """Scenario 3: Compliance Check.

        Query: "Is our approach compliant with the RFP requirements?"
        Expected: Compliance assessment with specific references.
        """
        return TestScenario(
            name="Compliance Check",
            description="Check if proposed approach meets RFP requirements",
            query_type=QueryType.COMPLIANCE_CHECK,
            request=AnalysisRequest(
                query="Does our authentication approach satisfy the RFP security requirements?",
                query_type=QueryType.COMPLIANCE_CHECK,
                additional_context={
                    "approach": "Multi-factor authentication using TOTP and biometrics",
                    "section": "Security Requirements",
                },
            ),
            expected_outputs={
                "has_compliance_assessment": True,
                "has_specific_references": True,
                "has_reasoning": True,
            },
            scoring_criteria={
                "compliance_accuracy": 30,
                "requirement_coverage": 25,
                "specific_references": 20,
                "reasoning_quality": 15,
                "format_correctness": 10,
            },
        )

    @staticmethod
    def scenario_4_conflict_detection() -> TestScenario:
        """Scenario 4: Conflict Detection.

        Query: "Are timeline, budget, and scope realistic together?"
        Expected: Identified conflicts or confirmation of consistency.
        """
        return TestScenario(
            name="Conflict Detection",
            description="Detect conflicts between timeline, budget, and scope",
            query_type=QueryType.CONFLICT_DETECTION,
            request=AnalysisRequest(
                query="Are the timeline, budget, and scope requirements realistic together?",
                query_type=QueryType.CONFLICT_DETECTION,
                additional_context={
                    "timeline": "6 months for full implementation",
                    "budget": "$500,000 total budget",
                    "scope": "Complete system overhaul with new features",
                },
            ),
            expected_outputs={
                "has_conflict_analysis": True,
                "considers_all_factors": True,
                "has_reasoning": True,
            },
            scoring_criteria={
                "conflict_identification": 30,
                "factor_coverage": 25,
                "analysis_depth": 20,
                "reasoning_quality": 15,
                "format_correctness": 10,
            },
        )

    @staticmethod
    def scenario_5_ambiguity_handling() -> TestScenario:
        """Scenario 5: Ambiguity Handling.

        Query: "What does 'adequate security' mean in this RFP?"
        Expected: Analysis of ambiguous terms with interpretations.
        """
        return TestScenario(
            name="Ambiguity Handling",
            description="Handle ambiguous terms and provide interpretations",
            query_type=QueryType.AMBIGUITY_ANALYSIS,
            request=AnalysisRequest(
                query="What does 'adequate security' mean in this RFP?",
                query_type=QueryType.AMBIGUITY_ANALYSIS,
                additional_context={
                    "term": "adequate security",
                },
            ),
            expected_outputs={
                "identifies_ambiguity": True,
                "provides_interpretations": True,
                "suggests_clarifications": True,
            },
            scoring_criteria={
                "ambiguity_recognition": 25,
                "interpretation_quality": 25,
                "clarification_suggestions": 20,
                "context_awareness": 20,
                "format_correctness": 10,
            },
        )

    @classmethod
    def get_all_scenarios(cls) -> list[TestScenario]:
        """Get all 5 test scenarios.

        Returns:
            List of all test scenarios.
        """
        return [
            cls.scenario_1_requirement_extraction(),
            cls.scenario_2_gap_analysis(),
            cls.scenario_3_compliance_check(),
            cls.scenario_4_conflict_detection(),
            cls.scenario_5_ambiguity_handling(),
        ]

    @classmethod
    def get_scenario_by_name(cls, name: str) -> TestScenario | None:
        """Get a specific scenario by name.

        Args:
            name: Scenario name.

        Returns:
            TestScenario or None if not found.
        """
        scenarios = {
            "requirement_extraction": cls.scenario_1_requirement_extraction,
            "gap_analysis": cls.scenario_2_gap_analysis,
            "compliance_check": cls.scenario_3_compliance_check,
            "conflict_detection": cls.scenario_4_conflict_detection,
            "ambiguity_handling": cls.scenario_5_ambiguity_handling,
        }
        factory = scenarios.get(name.lower().replace(" ", "_"))
        return factory() if factory else None
