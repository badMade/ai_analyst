"""Decision Maker module for Agentic AI.

Provides action selection, utility-based decision making,
and multi-criteria optimization.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import logging
import random

logger = logging.getLogger(__name__)


class DecisionStrategy(Enum):
    """Strategies for decision making."""
    UTILITY_MAXIMIZATION = "utility_maximization"
    SATISFICING = "satisficing"
    MINIMAX = "minimax"
    EXPECTED_UTILITY = "expected_utility"
    MULTI_CRITERIA = "multi_criteria"


@dataclass
class Option:
    """An option/action to choose from."""
    id: str
    name: str
    description: str = ""
    utility: float = 0.0
    probability: float = 1.0
    criteria_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """A decision made by the decision maker."""
    selected_option: Option
    alternatives: list[Option]
    strategy: DecisionStrategy
    confidence: float
    reasoning: str


@dataclass
class Criterion:
    """A criterion for multi-criteria decision making."""
    name: str
    weight: float = 1.0
    minimize: bool = False  # True if lower is better


class DecisionMaker:
    """Decision making engine for agents."""

    def __init__(
        self,
        default_strategy: DecisionStrategy = DecisionStrategy.UTILITY_MAXIMIZATION,
    ):
        """Initialize the decision maker.

        Args:
            default_strategy: Default strategy to use.
        """
        self.default_strategy = default_strategy
        self.criteria: list[Criterion] = []
        self.decision_history: list[Decision] = []
        self._utility_functions: dict[str, Callable] = {}

    def decide(
        self,
        options: list[Option],
        strategy: DecisionStrategy | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> Decision:
        """Make a decision from available options.

        Args:
            options: Available options to choose from.
            strategy: Strategy to use (defaults to default_strategy).
            constraints: Optional constraints on the decision.

        Returns:
            The decision made.
        """
        if not options:
            raise ValueError("No options provided for decision")

        strategy = strategy or self.default_strategy

        # Filter options by constraints
        valid_options = self._filter_by_constraints(options, constraints)
        if not valid_options:
            logger.warning("No valid options after applying constraints")
            valid_options = options

        # Apply decision strategy
        if strategy == DecisionStrategy.UTILITY_MAXIMIZATION:
            selected, reasoning = self._utility_maximization(valid_options)
        elif strategy == DecisionStrategy.SATISFICING:
            selected, reasoning = self._satisficing(valid_options, constraints)
        elif strategy == DecisionStrategy.MINIMAX:
            selected, reasoning = self._minimax(valid_options)
        elif strategy == DecisionStrategy.EXPECTED_UTILITY:
            selected, reasoning = self._expected_utility(valid_options)
        elif strategy == DecisionStrategy.MULTI_CRITERIA:
            selected, reasoning = self._multi_criteria(valid_options)
        else:
            selected, reasoning = self._utility_maximization(valid_options)

        # Calculate confidence
        confidence = self._calculate_confidence(selected, valid_options)

        decision = Decision(
            selected_option=selected,
            alternatives=[o for o in valid_options if o.id != selected.id],
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
        )

        self.decision_history.append(decision)
        logger.info(f"Decision made: {selected.name} (confidence: {confidence:.2f})")

        return decision

    def add_criterion(self, criterion: Criterion) -> None:
        """Add a criterion for multi-criteria decisions.

        Args:
            criterion: The criterion to add.
        """
        self.criteria.append(criterion)

    def set_criteria_weights(self, weights: dict[str, float]) -> None:
        """Set weights for existing criteria.

        Args:
            weights: Dictionary mapping criterion names to weights.
        """
        for criterion in self.criteria:
            if criterion.name in weights:
                criterion.weight = weights[criterion.name]

    def register_utility_function(
        self,
        name: str,
        func: Callable[[Option], float],
    ) -> None:
        """Register a custom utility function.

        Args:
            name: Function name.
            func: Utility calculation function.
        """
        self._utility_functions[name] = func

    def evaluate_option(self, option: Option) -> float:
        """Evaluate the utility of an option.

        Args:
            option: Option to evaluate.

        Returns:
            Utility value.
        """
        if self._utility_functions:
            # Use custom utility functions
            utilities = [func(option) for func in self._utility_functions.values()]
            return sum(utilities) / len(utilities)

        return option.utility

    def _filter_by_constraints(
        self,
        options: list[Option],
        constraints: dict[str, Any] | None,
    ) -> list[Option]:
        """Filter options by constraints."""
        if not constraints:
            return options

        valid = []
        for option in options:
            meets_constraints = True
            for key, value in constraints.items():
                if key == "min_utility" and option.utility < value:
                    meets_constraints = False
                elif key == "max_utility" and option.utility > value:
                    meets_constraints = False
                elif key == "required_criteria":
                    for criterion in value:
                        if criterion not in option.criteria_scores:
                            meets_constraints = False

            if meets_constraints:
                valid.append(option)

        return valid

    def _utility_maximization(
        self,
        options: list[Option],
    ) -> tuple[Option, str]:
        """Select option with highest utility."""
        best = max(options, key=lambda o: self.evaluate_option(o))
        reasoning = f"Selected {best.name} with highest utility: {best.utility:.2f}"
        return best, reasoning

    def _satisficing(
        self,
        options: list[Option],
        constraints: dict[str, Any] | None,
    ) -> tuple[Option, str]:
        """Select first option meeting threshold."""
        threshold = (constraints or {}).get("threshold", 0.5)

        for option in options:
            if self.evaluate_option(option) >= threshold:
                reasoning = f"Selected {option.name} as it meets threshold {threshold}"
                return option, reasoning

        # Fall back to best available
        best = max(options, key=lambda o: self.evaluate_option(o))
        reasoning = f"No option met threshold; selected best: {best.name}"
        return best, reasoning

    def _minimax(self, options: list[Option]) -> tuple[Option, str]:
        """Minimax decision - maximize minimum outcome."""
        # Assumes worst case for each option is stored in metadata
        def worst_case(o: Option) -> float:
            return o.metadata.get("worst_case", o.utility * 0.5)

        best = max(options, key=worst_case)
        reasoning = f"Selected {best.name} with best worst-case outcome"
        return best, reasoning

    def _expected_utility(self, options: list[Option]) -> tuple[Option, str]:
        """Select option with highest expected utility."""
        def expected(o: Option) -> float:
            return o.utility * o.probability

        best = max(options, key=expected)
        eu = best.utility * best.probability
        reasoning = f"Selected {best.name} with expected utility: {eu:.2f}"
        return best, reasoning

    def _multi_criteria(self, options: list[Option]) -> tuple[Option, str]:
        """Multi-criteria decision making."""
        if not self.criteria:
            return self._utility_maximization(options)

        def weighted_score(option: Option) -> float:
            score = 0.0
            total_weight = sum(c.weight for c in self.criteria)

            for criterion in self.criteria:
                if criterion.name in option.criteria_scores:
                    value = option.criteria_scores[criterion.name]
                    if criterion.minimize:
                        value = 1.0 - value  # Invert for minimization
                    score += value * criterion.weight

            return score / total_weight if total_weight > 0 else 0.0

        best = max(options, key=weighted_score)
        reasoning = f"Selected {best.name} using {len(self.criteria)} criteria"
        return best, reasoning

    def _calculate_confidence(
        self,
        selected: Option,
        options: list[Option],
    ) -> float:
        """Calculate confidence in the decision."""
        if len(options) == 1:
            return 1.0

        utilities = [self.evaluate_option(o) for o in options]
        selected_utility = self.evaluate_option(selected)

        # Confidence based on gap between selected and second-best
        sorted_utilities = sorted(utilities, reverse=True)
        if len(sorted_utilities) > 1:
            gap = sorted_utilities[0] - sorted_utilities[1]
            max_utility = max(abs(u) for u in utilities) or 1.0
            confidence = min(1.0, 0.5 + (gap / max_utility) * 0.5)
        else:
            confidence = 0.8

        return confidence
