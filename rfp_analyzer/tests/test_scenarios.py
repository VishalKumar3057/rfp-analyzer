"""Tests for evaluation scenarios."""

import pytest

from src.evaluation.scenarios import TestScenarios, TestScenario
from src.models.requests import QueryType


class TestTestScenarios:
    """Tests for TestScenarios class."""

    def test_get_all_scenarios(self):
        """Test getting all 5 scenarios."""
        scenarios = TestScenarios.get_all_scenarios()
        
        assert len(scenarios) == 5
        assert all(isinstance(s, TestScenario) for s in scenarios)

    def test_scenario_1_requirement_extraction(self):
        """Test requirement extraction scenario."""
        scenario = TestScenarios.scenario_1_requirement_extraction()
        
        assert scenario.name == "Requirement Extraction"
        assert scenario.query_type == QueryType.REQUIREMENT_EXTRACTION
        assert "requirements" in scenario.request.query.lower()
        assert scenario.max_score == 100

    def test_scenario_2_gap_analysis(self):
        """Test gap analysis scenario."""
        scenario = TestScenarios.scenario_2_gap_analysis()
        
        assert scenario.name == "Gap Analysis"
        assert scenario.query_type == QueryType.GAP_ANALYSIS
        assert "missing" in scenario.request.query.lower()

    def test_scenario_3_compliance_check(self):
        """Test compliance check scenario."""
        scenario = TestScenarios.scenario_3_compliance_check()
        
        assert scenario.name == "Compliance Check"
        assert scenario.query_type == QueryType.COMPLIANCE_CHECK
        assert scenario.request.additional_context.get("approach") is not None

    def test_scenario_4_conflict_detection(self):
        """Test conflict detection scenario."""
        scenario = TestScenarios.scenario_4_conflict_detection()
        
        assert scenario.name == "Conflict Detection"
        assert scenario.query_type == QueryType.CONFLICT_DETECTION
        assert "timeline" in scenario.request.additional_context
        assert "budget" in scenario.request.additional_context

    def test_scenario_5_ambiguity_handling(self):
        """Test ambiguity handling scenario."""
        scenario = TestScenarios.scenario_5_ambiguity_handling()
        
        assert scenario.name == "Ambiguity Handling"
        assert scenario.query_type == QueryType.AMBIGUITY_ANALYSIS
        assert "term" in scenario.request.additional_context

    def test_scoring_criteria_sum(self):
        """Test that scoring criteria sum to 100 for each scenario."""
        scenarios = TestScenarios.get_all_scenarios()
        
        for scenario in scenarios:
            total = sum(scenario.scoring_criteria.values())
            assert total == 100, f"{scenario.name} criteria sum to {total}, not 100"

    def test_get_scenario_by_name(self):
        """Test getting scenario by name."""
        scenario = TestScenarios.get_scenario_by_name("requirement_extraction")
        assert scenario is not None
        assert scenario.name == "Requirement Extraction"

        scenario = TestScenarios.get_scenario_by_name("gap_analysis")
        assert scenario is not None
        assert scenario.name == "Gap Analysis"

    def test_get_scenario_by_name_not_found(self):
        """Test getting non-existent scenario."""
        scenario = TestScenarios.get_scenario_by_name("nonexistent")
        assert scenario is None

    def test_expected_outputs_defined(self):
        """Test that all scenarios have expected outputs."""
        scenarios = TestScenarios.get_all_scenarios()
        
        for scenario in scenarios:
            assert scenario.expected_outputs is not None
            assert len(scenario.expected_outputs) > 0

    def test_scenario_requests_valid(self):
        """Test that all scenario requests are valid."""
        scenarios = TestScenarios.get_all_scenarios()
        
        for scenario in scenarios:
            request = scenario.request
            assert request.query is not None
            assert len(request.query) >= 5
            assert request.query_type is not None

