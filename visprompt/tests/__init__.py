"""Auto-generated visual unit test framework.

Provides a standalone test runner that can execute Critic-generated tests
against evaluation results, independent of the LLM. This enables:
1. Reproducible test execution (same tests, same results)
2. Comparison between auto-generated and hand-written tests
3. Test suite analysis across iterations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from visprompt.utils.metrics import EvalResult

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single visual unit test."""

    name: str
    category: str  # "accuracy", "balance", "confusion", "calibration", "efficiency"
    check_fn: Callable[[EvalResult], tuple[str, Any, Any]]
    # check_fn returns (status, actual_value, threshold)
    description: str = ""

    def run(self, eval_result: EvalResult) -> dict:
        try:
            status, actual, threshold = self.check_fn(eval_result)
            return {
                "name": self.name,
                "category": self.category,
                "status": status,
                "actual_value": actual,
                "threshold": threshold,
                "description": self.description,
            }
        except Exception as e:
            return {
                "name": self.name,
                "category": self.category,
                "status": "ERROR",
                "actual_value": str(e),
                "threshold": None,
                "description": self.description,
            }


@dataclass
class TestSuite:
    """Collection of visual unit tests."""

    tests: list[TestCase] = field(default_factory=list)
    name: str = "auto_generated"

    def add(self, test: TestCase) -> None:
        self.tests.append(test)

    def run_all(self, eval_result: EvalResult) -> dict:
        results = []
        for test in self.tests:
            results.append(test.run(eval_result))

        n_pass = sum(1 for r in results if r["status"] == "PASS")
        n_fail = sum(1 for r in results if r["status"] == "FAIL")
        n_warn = sum(1 for r in results if r["status"] == "WARNING")

        verdict = "PASS"
        if n_fail > 0:
            verdict = "NEEDS_REFINEMENT"
        elif n_warn > 1:
            verdict = "ACCEPTABLE"

        return {
            "suite_name": self.name,
            "n_tests": len(self.tests),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_warning": n_warn,
            "verdict": verdict,
            "results": results,
        }


class TestGenerator:
    """Generates test suites programmatically based on task type.

    This is the DETERMINISTIC counterpart to the LLM-based Critic.
    Used for:
    - Ablation: comparing auto-generated vs programmatic tests
    - Reproducibility: same tests every run
    - Validation: ensuring LLM-generated tests cover key areas
    """

    @staticmethod
    def for_classification(
        n_classes: int,
        random_acc: Optional[float] = None,
        min_class_threshold: float = 0.5,
    ) -> TestSuite:
        """Generate classification-specific test suite."""
        suite = TestSuite(name="classification_auto")
        if random_acc is None:
            random_acc = 1.0 / max(n_classes, 1)

        # Test 1: Overall accuracy above random
        suite.add(TestCase(
            name="overall_accuracy",
            category="accuracy",
            description=f"Overall accuracy should be well above random ({random_acc:.2%})",
            check_fn=lambda r: (
                "PASS" if r.primary_metric > random_acc * 5 else "FAIL",
                r.primary_metric,
                random_acc * 5,
            ),
        ))

        # Test 2: No class near random
        suite.add(TestCase(
            name="per_class_minimum",
            category="balance",
            description=f"No class should be below {min_class_threshold:.0%}",
            check_fn=lambda r: _check_class_minimum(r, min_class_threshold),
        ))

        # Test 3: Class balance (max gap)
        suite.add(TestCase(
            name="class_balance",
            category="balance",
            description="Gap between best and worst class should be < 50%",
            check_fn=lambda r: _check_class_balance(r, 0.50),
        ))

        # Test 4: Symmetric confusion detection
        suite.add(TestCase(
            name="symmetric_confusion",
            category="confusion",
            description="Check for symmetric confusion pairs (fundamental limits)",
            check_fn=_check_symmetric_confusion,
        ))

        # Test 5: Confidence calibration
        suite.add(TestCase(
            name="confidence_separation",
            category="calibration",
            description="Correct predictions should have higher confidence than wrong ones",
            check_fn=_check_confidence_separation,
        ))

        # Test 6: Ensemble value (if per-level data available)
        suite.add(TestCase(
            name="positive_contribution",
            category="efficiency",
            description="Accuracy should improve over single-template baseline",
            check_fn=lambda r: (
                "PASS" if r.primary_metric > random_acc * 10 else "FAIL",
                r.primary_metric,
                random_acc * 10,
            ),
        ))

        return suite

    @staticmethod
    def for_segmentation(
        min_iou: float = 0.5,
        min_boundary_f1: float = 0.4,
    ) -> TestSuite:
        """Generate segmentation-specific test suite."""
        suite = TestSuite(name="segmentation_auto")

        suite.add(TestCase(
            name="mean_iou_threshold",
            category="accuracy",
            description=f"Mean IoU should exceed {min_iou}",
            check_fn=lambda r: (
                "PASS" if r.primary_metric > min_iou else "FAIL",
                r.primary_metric,
                min_iou,
            ),
        ))

        suite.add(TestCase(
            name="per_sample_minimum",
            category="balance",
            description="No sample should have IoU below 0.1",
            check_fn=lambda r: _check_class_minimum(r, 0.1),
        ))

        suite.add(TestCase(
            name="size_stratified",
            category="balance",
            description="IoU should not drop drastically for small objects",
            check_fn=lambda r: (
                "INFO",
                r.class_accuracy_stats().get("std", 0),
                "std < 0.3",
            ),
        ))

        return suite

    @staticmethod
    def for_detection(
        min_ap: float = 0.2,
        n_classes: int = 100,
    ) -> TestSuite:
        """Generate detection-specific test suite."""
        suite = TestSuite(name="detection_auto")

        suite.add(TestCase(
            name="mean_ap_threshold",
            category="accuracy",
            description=f"mAP should exceed {min_ap}",
            check_fn=lambda r: (
                "PASS" if r.primary_metric > min_ap else "FAIL",
                r.primary_metric,
                min_ap,
            ),
        ))

        suite.add(TestCase(
            name="rare_class_recall",
            category="balance",
            description="Rare classes should have non-zero AP",
            check_fn=lambda r: (
                "PASS" if all(v > 0 for v in r.per_class_metrics.values()) else "WARNING",
                sum(1 for v in r.per_class_metrics.values() if v == 0),
                "0 classes with AP=0",
            ),
        ))

        suite.add(TestCase(
            name="class_coverage",
            category="balance",
            description="At least 80% of classes should be detected at least once",
            check_fn=lambda r: (
                "PASS" if sum(1 for v in r.per_class_metrics.values() if v > 0) / max(len(r.per_class_metrics), 1) > 0.8 else "FAIL",
                sum(1 for v in r.per_class_metrics.values() if v > 0) / max(len(r.per_class_metrics), 1),
                0.8,
            ),
        ))

        return suite


# ── Helper check functions ────────────────────────────────────────────────────

def _check_class_minimum(
    result: EvalResult, threshold: float
) -> tuple[str, Any, Any]:
    if not result.per_class_metrics:
        return "INFO", "no per-class data", threshold
    below = [
        (cls, val) for cls, val in result.per_class_metrics.items()
        if val < threshold
    ]
    if below:
        return "FAIL", f"{len(below)} classes below {threshold}", threshold
    return "PASS", "all above threshold", threshold


def _check_class_balance(
    result: EvalResult, max_gap: float
) -> tuple[str, Any, Any]:
    if not result.per_class_metrics:
        return "INFO", "no data", max_gap
    vals = list(result.per_class_metrics.values())
    gap = max(vals) - min(vals)
    status = "PASS" if gap < max_gap else "FAIL"
    return status, round(gap, 4), max_gap


def _check_symmetric_confusion(
    result: EvalResult,
) -> tuple[str, Any, Any]:
    pairs = result.confusion_pairs(5)
    if not pairs:
        return "INFO", "no confusion data", "N/A"

    symmetric = []
    pair_set = {(a, b): c for a, b, c in pairs}
    for (a, b), count in pair_set.items():
        if (b, a) in pair_set:
            symmetric.append(f"{a}↔{b}")

    if symmetric:
        return "WARNING", f"Symmetric: {', '.join(symmetric[:3])}", "possible fundamental limit"
    return "PASS", "no symmetric confusion", "N/A"


def _check_confidence_separation(
    result: EvalResult,
) -> tuple[str, Any, Any]:
    if result.confidence_scores is None or result.predictions is None or result.ground_truth is None:
        return "INFO", "no confidence data", "N/A"

    correct = result.predictions == result.ground_truth
    if not correct.any() or correct.all():
        return "INFO", "insufficient data", "N/A"

    correct_mean = float(result.confidence_scores[correct].mean())
    wrong_mean = float(result.confidence_scores[~correct].mean())
    ratio = correct_mean / max(wrong_mean, 1e-8)

    status = "PASS" if ratio > 1.15 else ("WARNING" if ratio > 1.0 else "FAIL")
    return status, round(ratio, 3), 1.15
