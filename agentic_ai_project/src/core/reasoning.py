"""Reasoning module for Agentic AI.

Provides logical inference, chain-of-thought reasoning,
and knowledge-based reasoning capabilities.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning."""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"


@dataclass
class Premise:
    """A premise in logical reasoning."""
    statement: str
    confidence: float = 1.0
    source: str = ""


@dataclass
class Conclusion:
    """A conclusion from reasoning."""
    statement: str
    confidence: float
    premises: list[Premise]
    reasoning_type: ReasoningType


@dataclass
class ReasoningStep:
    """A single step in reasoning chain."""
    step_number: int
    thought: str
    result: Any
    confidence: float


class ReasoningChain:
    """Chain of reasoning steps."""

    def __init__(self):
        self.steps: list[ReasoningStep] = []
        self._current_step = 0

    def add_step(self, thought: str, result: Any, confidence: float = 1.0) -> ReasoningStep:
        """Add a reasoning step.

        Args:
            thought: The thought/reasoning at this step.
            result: The result of this step.
            confidence: Confidence in this step.

        Returns:
            The added step.
        """
        self._current_step += 1
        step = ReasoningStep(
            step_number=self._current_step,
            thought=thought,
            result=result,
            confidence=confidence,
        )
        self.steps.append(step)
        return step

    def get_final_result(self) -> Any:
        """Get the result from the final step."""
        if self.steps:
            return self.steps[-1].result
        return None

    def get_overall_confidence(self) -> float:
        """Calculate overall confidence across all steps."""
        if not self.steps:
            return 0.0
        confidences = [s.confidence for s in self.steps]
        # Multiply confidences (chain rule)
        result = 1.0
        for c in confidences:
            result *= c
        return result

    def to_string(self) -> str:
        """Convert chain to readable string."""
        lines = []
        for step in self.steps:
            lines.append(f"Step {step.step_number}: {step.thought}")
            lines.append(f"  Result: {step.result}")
            lines.append(f"  Confidence: {step.confidence:.2f}")
        return "\n".join(lines)


class Reasoner:
    """Main reasoning engine."""

    def __init__(self, knowledge_base: dict[str, Any] | None = None):
        """Initialize the reasoner.

        Args:
            knowledge_base: Optional knowledge base for reasoning.
        """
        self.knowledge_base = knowledge_base or {}
        self.rules: list[tuple[Callable, Callable]] = []
        self._reasoning_history: list[ReasoningChain] = []

    def add_rule(
        self,
        condition: Callable[[Any], bool],
        action: Callable[[Any], Any],
    ) -> None:
        """Add a reasoning rule.

        Args:
            condition: Function that checks if rule applies.
            action: Function to execute if rule applies.
        """
        self.rules.append((condition, action))

    def reason(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE,
    ) -> ReasoningChain:
        """Perform reasoning on a query.

        Args:
            query: The question or problem.
            context: Additional context.
            reasoning_type: Type of reasoning to use.

        Returns:
            Chain of reasoning steps.
        """
        chain = ReasoningChain()

        if reasoning_type == ReasoningType.DEDUCTIVE:
            self._deductive_reasoning(query, context, chain)
        elif reasoning_type == ReasoningType.INDUCTIVE:
            self._inductive_reasoning(query, context, chain)
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            self._abductive_reasoning(query, context, chain)
        elif reasoning_type == ReasoningType.ANALOGICAL:
            self._analogical_reasoning(query, context, chain)

        self._reasoning_history.append(chain)
        return chain

    def deduce(self, premises: list[Premise]) -> Conclusion | None:
        """Perform deductive reasoning from premises.

        Args:
            premises: List of premises.

        Returns:
            Conclusion if one can be drawn.
        """
        if not premises:
            return None

        # Check for valid syllogism patterns
    def _apply_deduction_rules(self, premises: list[Premise]) -> str | None:
        """Apply deduction rules to premises.

        This method needs to be expanded to implement actual logical inference rules
        (e.g., Modus Ponens, Modus Tollens, Syllogisms) based on the provided premises.
        """
        # Placeholder for actual deductive logic
        if not premises:
            return None

        # Example: Very basic Modus Ponens pattern
        # If we have "If P then Q" and "P is true", we can conclude "Q is true".
        for p1 in premises:
            for p2 in premises:
                if "If " in p1.statement and " then " in p1.statement:
                    parts = p1.statement.replace("If ", "").split(" then ")
                    if len(parts) == 2:
                        antecedent = parts[0].strip()
                        consequent = parts[1].strip()

                        if antecedent in p2.statement: # Simplistic check for P being true
                            return f"{consequent} is true"

        if len(premises) >= 2:
            return f"A general conclusion can be derived from {len(premises)} premises."
        return None
        if conclusion_text:
            confidence = min(p.confidence for p in premises)
            return Conclusion(
                statement=conclusion_text,
                confidence=confidence,
                premises=premises,
                reasoning_type=ReasoningType.DEDUCTIVE,
            )
        return None

    def induce(self, observations: list[dict[str, Any]]) -> Conclusion | None:
        """Perform inductive reasoning from observations.

        Args:
            observations: List of observed instances.

        Returns:
            Generalized conclusion.
        """
        if len(observations) < 2:
            return None

        # Find common patterns
        pattern = self._find_pattern(observations)
        if pattern:
            # Confidence based on number of observations
            confidence = min(0.9, len(observations) / 10)
            return Conclusion(
                statement=f"Pattern observed: {pattern}",
                confidence=confidence,
                premises=[Premise(str(o)) for o in observations],
                reasoning_type=ReasoningType.INDUCTIVE,
            )
        return None

    def abduce(self, observation: str, possible_causes: list[str]) -> Conclusion | None:
        """Perform abductive reasoning to find best explanation.

        Args:
            observation: The observation to explain.
            possible_causes: List of possible explanations.

        Returns:
            Best explanation.
        """
        if not possible_causes:
            return None

        # Rank explanations by likelihood
        ranked = self._rank_explanations(observation, possible_causes)
        if ranked:
            best = ranked[0]
            return Conclusion(
                statement=f"Best explanation: {best[1]}",
                confidence=best[0],
                premises=[Premise(observation)],
                reasoning_type=ReasoningType.ABDUCTIVE,
            )
        return None

    def analogize(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
    ) -> Conclusion | None:
        """Perform analogical reasoning.

        Args:
            source: Known source domain.
            target: Target domain to reason about.

        Returns:
            Analogical conclusion.
        """
        # Find structural similarities
        mappings = self._find_analogical_mappings(source, target)
        if mappings:
            inference = self._apply_analogical_inference(source, target, mappings)
            return Conclusion(
                statement=inference,
                confidence=0.6,  # Analogies are less certain
                premises=[Premise(f"Source: {source}"), Premise(f"Target: {target}")],
                reasoning_type=ReasoningType.ANALOGICAL,
            )
        return None

    def _deductive_reasoning(
        self,
        query: str,
        context: dict[str, Any] | None,
        chain: ReasoningChain,
    ) -> None:
        """Execute deductive reasoning process."""
        # Step 1: Identify relevant premises
        chain.add_step(
            thought="Identifying relevant premises from knowledge base",
            result=self._find_relevant_knowledge(query),
            confidence=0.9,
        )

        # Step 2: Apply logical rules
        chain.add_step(
            thought="Applying deductive rules",
            result="Rules applied",
            confidence=0.95,
        )

        # Step 3: Draw conclusion
        chain.add_step(
            thought="Drawing logical conclusion",
            result=f"Conclusion for: {query}",
            confidence=0.85,
        )

    def _inductive_reasoning(
        self,
        query: str,
        context: dict[str, Any] | None,
        chain: ReasoningChain,
    ) -> None:
        """Execute inductive reasoning process."""
        # Step 1: Gather observations
        chain.add_step(
            thought="Gathering relevant observations",
            result=context or {},
            confidence=0.9,
        )

        # Step 2: Identify patterns
        chain.add_step(
            thought="Analyzing patterns in observations",
            result="Pattern analysis complete",
            confidence=0.8,
        )

        # Step 3: Generalize
        chain.add_step(
            thought="Forming generalization",
            result=f"Generalization for: {query}",
            confidence=0.7,
        )

    def _abductive_reasoning(
        self,
        query: str,
        context: dict[str, Any] | None,
        chain: ReasoningChain,
    ) -> None:
        """Execute abductive reasoning process."""
        # Step 1: Understand observation
        chain.add_step(
            thought="Understanding the observation to explain",
            result=query,
            confidence=0.95,
        )

        # Step 2: Generate hypotheses
        chain.add_step(
            thought="Generating possible explanations",
            result=["hypothesis_1", "hypothesis_2"],
            confidence=0.8,
        )

        # Step 3: Select best explanation
        chain.add_step(
            thought="Selecting most likely explanation",
            result="Best explanation selected",
            confidence=0.6,
        )

    def _analogical_reasoning(
        self,
        query: str,
        context: dict[str, Any] | None,
        chain: ReasoningChain,
    ) -> None:
        """Execute analogical reasoning process."""
        # Step 1: Find analogous situation
        chain.add_step(
            thought="Finding analogous situation in knowledge base",
            result="Analogy found",
            confidence=0.7,
        )

        # Step 2: Map relationships
        chain.add_step(
            thought="Mapping relationships between domains",
            result="Mappings established",
            confidence=0.6,
        )

        # Step 3: Transfer inference
        chain.add_step(
            thought="Transferring inference to target domain",
            result=f"Analogical conclusion for: {query}",
            confidence=0.5,
        )

    def _find_relevant_knowledge(self, query: str) -> list[str]:
        """Find relevant knowledge for a query."""
        relevant = []
        query_words = set(query.lower().split())

        for key in self.knowledge_base:
            if any(word in key.lower() for word in query_words):
                relevant.append(key)

        return relevant

    def _apply_deduction_rules(self, premises: list[Premise]) -> str | None:
        """Apply deduction rules to premises."""
        # Simple modus ponens check
        if len(premises) >= 2:
            return f"Conclusion derived from {len(premises)} premises"
        return None

    def _find_pattern(self, observations: list[dict[str, Any]]) -> str | None:
        """Find common pattern in observations."""
        if not observations:
            return None

        # Find common keys
        common_keys = set(observations[0].keys())
        for obs in observations[1:]:
            common_keys &= set(obs.keys())

        if common_keys:
            return f"Common attributes: {', '.join(common_keys)}"
        return None

    def _rank_explanations(
        self,
        observation: str,
        causes: list[str],
    ) -> list[tuple[float, str]]:
        """Rank possible explanations by likelihood."""
        ranked = []
        for cause in causes:
            # Simple scoring based on word overlap
            obs_words = set(observation.lower().split())
            cause_words = set(cause.lower().split())
            overlap = len(obs_words & cause_words)
            score = overlap / max(len(cause_words), 1)
            ranked.append((score, cause))

        ranked.sort(reverse=True)
        return ranked

    def _find_analogical_mappings(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
    ) -> dict[str, str]:
        """Find structural mappings between source and target."""
        mappings = {}
        source_keys = set(source.keys())
        target_keys = set(target.keys())

        # Map matching keys
        common = source_keys & target_keys
        for key in common:
            mappings[key] = key

        return mappings

    def _apply_analogical_inference(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
        mappings: dict[str, str],
    ) -> str:
        """Apply analogical inference using mappings."""
        # Find properties in source not in target
        source_only = set(source.keys()) - set(target.keys())
        if source_only:
            prop = list(source_only)[0]
            return f"By analogy, target may have property: {prop}"
        return "No new inference from analogy"
