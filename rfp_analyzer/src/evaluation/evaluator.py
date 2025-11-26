"""RFP Analysis Evaluator."""

from typing import Any

from langsmith import traceable

from src.evaluation.metrics import (
    EvaluationMetrics,
    EvaluationResult,
    ScenarioScore,
    calculate_requirement_score,
)
from src.evaluation.scenarios import TestScenario, TestScenarios
from src.graph.workflow import RFPAnalysisGraph
from src.models.responses import AnalysisResponse
from src.utils.logging import LoggerMixin


class RFPEvaluator(LoggerMixin):
    """Evaluator for RFP analysis system."""

    PASS_THRESHOLD = 80  # Minimum score to pass

    def __init__(self, graph: RFPAnalysisGraph):
        """Initialize evaluator.

        Args:
            graph: RFP analysis graph to evaluate.
        """
        self._graph = graph

    @traceable(name="evaluate_all_scenarios")
    def evaluate_all(self) -> EvaluationResult:
        """Run all 5 test scenarios and evaluate.

        Returns:
            Overall evaluation result.
        """
        self.log_info("Starting evaluation of all scenarios")

        scenarios = TestScenarios.get_all_scenarios()
        scenario_scores = []
        total_score = 0.0

        for scenario in scenarios:
            score = self.evaluate_scenario(scenario)
            scenario_scores.append(score)
            total_score += score.total_score

        # Calculate overall results
        avg_score = total_score / len(scenarios) if scenarios else 0
        passed = sum(1 for s in scenario_scores if s.passed)
        failed = len(scenarios) - passed

        result = EvaluationResult(
            total_score=avg_score,
            scenario_scores=scenario_scores,
            passed_scenarios=passed,
            failed_scenarios=failed,
            overall_feedback=self._generate_overall_feedback(avg_score, passed, failed),
        )

        self.log_info(
            "Evaluation complete",
            avg_score=avg_score,
            passed=passed,
            failed=failed,
        )

        return result

    @traceable(name="evaluate_scenario")
    def evaluate_scenario(self, scenario: TestScenario) -> ScenarioScore:
        """Evaluate a single test scenario.

        Args:
            scenario: Test scenario to evaluate.

        Returns:
            Scenario score.
        """
        self.log_info("Evaluating scenario", name=scenario.name)

        try:
            # Run the analysis
            response = self._graph.run(scenario.request)

            # Calculate metrics
            metrics = EvaluationMetrics.from_response(response)

            # Score the response
            score, feedback = self._score_response(
                response=response,
                scenario=scenario,
            )

            passed = score >= self.PASS_THRESHOLD

            return ScenarioScore(
                scenario_name=scenario.name,
                total_score=score,
                component_scores=scenario.scoring_criteria,
                passed=passed,
                feedback=feedback,
                metrics=metrics,
            )

        except Exception as e:
            self.log_error("Scenario evaluation failed", error=str(e))
            return ScenarioScore(
                scenario_name=scenario.name,
                total_score=0,
                passed=False,
                feedback=[f"Error: {str(e)}"],
            )

    def _score_response(
        self,
        response: AnalysisResponse,
        scenario: TestScenario,
    ) -> tuple[float, list[str]]:
        """Score a response against scenario criteria.

        Args:
            response: Analysis response.
            scenario: Test scenario.

        Returns:
            Tuple of (score, feedback list).
        """
        # Use specialized scoring based on query type
        if scenario.name == "Requirement Extraction":
            return calculate_requirement_score(
                response, scenario.expected_outputs, scenario.scoring_criteria
            )

        # Generic scoring for other scenarios
        return self._generic_score(response, scenario)

    def _generic_score(
        self,
        response: AnalysisResponse,
        scenario: TestScenario,
    ) -> tuple[float, list[str]]:
        """Generic scoring for scenarios."""
        score = 0.0
        feedback = []
        criteria = scenario.scoring_criteria
        expected = scenario.expected_outputs

        # Check for requirements/findings
        if response.extracted_requirements:
            score += sum(criteria.values()) * 0.3
            feedback.append(f"✓ Found {len(response.extracted_requirements)} items")
        else:
            feedback.append("△ No specific items extracted")

        # Check for reasoning
        if response.reasoning and len(response.reasoning) > 100:
            score += sum(criteria.values()) * 0.3
            feedback.append("✓ Detailed reasoning provided")
        elif response.reasoning:
            score += sum(criteria.values()) * 0.15
            feedback.append("△ Brief reasoning provided")

        # Check for gaps/conflicts
        if expected.get("has_gaps") or expected.get("has_conflict_analysis"):
            if response.gaps_or_conflicts:
                score += sum(criteria.values()) * 0.2
                feedback.append(f"✓ Identified {len(response.gaps_or_conflicts)} issues")

        # Check confidence
        if response.confidence >= 70:
            score += sum(criteria.values()) * 0.1
            feedback.append(f"✓ High confidence: {response.confidence}%")

        # Format correctness
        score += sum(criteria.values()) * 0.1
        feedback.append("✓ Valid response format")

        return min(score, 100), feedback

    def _generate_overall_feedback(
        self, avg_score: float, passed: int, failed: int
    ) -> str:
        """Generate overall feedback message."""
        if avg_score >= 90:
            return f"Excellent! {passed}/5 scenarios passed with avg score {avg_score:.1f}"
        elif avg_score >= 80:
            return f"Good performance. {passed}/5 scenarios passed with avg score {avg_score:.1f}"
        elif avg_score >= 60:
            return f"Needs improvement. {passed}/5 scenarios passed with avg score {avg_score:.1f}"
        else:
            return f"Significant improvements needed. {passed}/5 passed, avg score {avg_score:.1f}"
