"""Reasoning Agent implementation.

An agent with advanced reasoning capabilities including chain-of-thought,
knowledge retrieval, and multi-step inference.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import logging

from .base_agent import BaseAgent, AgentConfig, AgentAction

logger = logging.getLogger(__name__)


class InferenceMethod(Enum):
    """Methods for reasoning inference."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    GRAPH_REASONING = "graph_reasoning"
    ANALOGICAL = "analogical"


@dataclass
class ReasoningConfig(AgentConfig):
    """Configuration for reasoning agents."""
    reasoning_depth: int = 5
    inference_method: str = "chain_of_thought"
    knowledge_retrieval: bool = True
    max_reasoning_time: float = 60.0
    confidence_threshold: float = 0.7


@dataclass
class ThoughtNode:
    """A node in the reasoning chain."""
    content: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    children: list["ThoughtNode"] = field(default_factory=list)
    parent: "ThoughtNode | None" = None


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning process."""
    query: str
    thoughts: list[ThoughtNode]
    conclusion: str
    confidence: float
    reasoning_steps: int


class ReasoningAgent(BaseAgent):
    """Agent with advanced reasoning capabilities.

    Implements various reasoning strategies including chain-of-thought,
    knowledge retrieval, and multi-step logical inference.
    """

    def __init__(
        self,
        name: str | None = None,
        config: ReasoningConfig | None = None,
        knowledge_base: dict[str, Any] | None = None,
    ):
        """Initialize the reasoning agent.

        Args:
            name: Optional name for the agent.
            config: Configuration settings.
            knowledge_base: Optional knowledge base for retrieval.
        """
        super().__init__(name, config or ReasoningConfig())
        self.config: ReasoningConfig = self.config
        self.knowledge_base = knowledge_base or {}
        self.reasoning_history: list[ReasoningTrace] = []
        self._current_trace: ReasoningTrace | None = None

    def perceive(self, observation: Any) -> dict[str, Any]:
        """Analyze observation and extract key information.

        Args:
            observation: Raw observation or query.

        Returns:
            Structured perception with context.
        """
        # Parse the observation
        query = self._extract_query(observation)
        context = self._gather_context(query)
        relevant_knowledge = self._retrieve_knowledge(query)

        return {
            "query": query,
            "context": context,
            "knowledge": relevant_knowledge,
            "requires_reasoning": self._requires_deep_reasoning(query),
        }

    def decide(self, perception: dict[str, Any]) -> AgentAction:
        """Reason about the perception and decide on action.

        Args:
            perception: Processed perception data.

        Returns:
            Reasoned action.
        """
        query = perception.get("query", "")
        knowledge = perception.get("knowledge", [])

        # Perform reasoning
        trace = self.reason(query, knowledge)

        return AgentAction(
            action_type="reasoned_response",
            parameters={
                "conclusion": trace.conclusion,
                "confidence": trace.confidence,
                "reasoning_steps": trace.reasoning_steps,
            }
        )

    def act(self, action: AgentAction) -> Any:
        """Execute the reasoned action.

        Args:
            action: Action containing reasoning results.

        Returns:
            Action execution result.
        """
        return {
            "response": action.parameters.get("conclusion"),
            "confidence": action.parameters.get("confidence"),
            "reasoning_steps": action.parameters.get("reasoning_steps"),
        }

    def reason(
        self,
        query: str,
        context: list[str] | None = None,
    ) -> ReasoningTrace:
        """Perform reasoning on a query.

        Args:
            query: The question or problem to reason about.
            context: Optional context information.

        Returns:
            Complete reasoning trace.
        """
        method = InferenceMethod(self.config.inference_method)

        if method == InferenceMethod.CHAIN_OF_THOUGHT:
            trace = self._chain_of_thought(query, context)
        elif method == InferenceMethod.TREE_OF_THOUGHT:
            trace = self._tree_of_thought(query, context)
        else:
            trace = self._chain_of_thought(query, context)

        self.reasoning_history.append(trace)
        return trace

    def add_knowledge(self, key: str, value: Any) -> None:
        """Add knowledge to the knowledge base.

        Args:
            key: Knowledge identifier.
            value: Knowledge content.
        """
        self.knowledge_base[key] = value
        logger.debug(f"{self.name}: Added knowledge - {key}")

    def query_knowledge(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Query the knowledge base.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            Relevant knowledge entries.
        """
        results = self._retrieve_knowledge(query)
        return results[:top_k]

    def _extract_query(self, observation: Any) -> str:
        """Extract query from observation."""
        if isinstance(observation, str):
            return observation
        if isinstance(observation, dict):
            return observation.get("query", str(observation))
        return str(observation)

    def _gather_context(self, query: str) -> list[str]:
        """Gather relevant context for query."""
        context = []
        # Add recent reasoning history
        for trace in self.reasoning_history[-3:]:
            if self._is_relevant(trace.query, query):
                context.append(f"Previous: {trace.conclusion}")
        return context

    def _retrieve_knowledge(self, query: str) -> list[dict[str, Any]]:
        """Retrieve relevant knowledge."""
        if not self.config.knowledge_retrieval:
            return []

        results = []
        query_lower = query.lower()

        for key, value in self.knowledge_base.items():
            if any(word in key.lower() for word in query_lower.split()):
                results.append({"key": key, "value": value, "relevance": 0.8})

        return sorted(results, key=lambda x: x["relevance"], reverse=True)

    def _requires_deep_reasoning(self, query: str) -> bool:
        """Determine if query requires deep reasoning."""
        complex_indicators = ["why", "how", "explain", "compare", "analyze", "evaluate"]
        return any(ind in query.lower() for ind in complex_indicators)

    def _chain_of_thought(
        self,
        query: str,
        context: list[str] | None,
    ) -> ReasoningTrace:
        """Perform chain-of-thought reasoning."""
        thoughts = []

        # Step 1: Understand the problem
        thought1 = ThoughtNode(
            content=f"Understanding query: {query}",
            confidence=0.95,
            evidence=context or [],
        )
        thoughts.append(thought1)

        # Step 2: Break down into sub-problems
        thought2 = ThoughtNode(
            content="Breaking down into components",
            confidence=0.9,
            parent=thought1,
        )
        thought1.children.append(thought2)
        thoughts.append(thought2)

        # Step 3: Apply knowledge
        thought3 = ThoughtNode(
            content="Applying relevant knowledge",
            confidence=0.85,
            parent=thought2,
        )
        thought2.children.append(thought3)
        thoughts.append(thought3)

        # Step 4: Synthesize conclusion
        conclusion = self._synthesize_conclusion(thoughts)
        final_confidence = min(t.confidence for t in thoughts)

        return ReasoningTrace(
            query=query,
            thoughts=thoughts,
            conclusion=conclusion,
            confidence=final_confidence,
            reasoning_steps=len(thoughts),
        )

    def _tree_of_thought(
        self,
        query: str,
        context: list[str] | None,
    ) -> ReasoningTrace:
        """Perform tree-of-thought reasoning with exploration."""
        root = ThoughtNode(
            content=f"Root: {query}",
            confidence=1.0,
        )

        # Generate multiple reasoning branches
        branches = self._generate_branches(root, depth=self.config.reasoning_depth)

        # Evaluate and select best path
        best_path = self._select_best_path(branches)

        conclusion = self._synthesize_conclusion(best_path)

        return ReasoningTrace(
            query=query,
            thoughts=best_path,
            conclusion=conclusion,
            confidence=min(t.confidence for t in best_path),
            reasoning_steps=len(best_path),
        )

    def _generate_branches(
        self,
        node: ThoughtNode,
        depth: int,
    ) -> list[ThoughtNode]:
        """Generate reasoning branches from a node."""
        if depth <= 0:
            return [node]

        branches = []
        # Generate 2-3 possible next thoughts
        for i in range(self.config.branching_factor):
            child = ThoughtNode(
                content=f"Branch {i+1} from: {node.content[:30]}...",
                confidence=node.confidence * 0.95,
                parent=node,
            )
            node.children.append(child)
            branches.extend(self._generate_branches(child, depth - 1))

        return [node] + branches

    def _select_best_path(self, nodes: list[ThoughtNode]) -> list[ThoughtNode]:
        """Select the best reasoning path."""
        # Sort by confidence and return the path to the best leaf
        best_node = max(nodes, key=lambda n: n.confidence)

        path = []
        current = best_node
        while current is not None:
            path.insert(0, current)
            current = current.parent

        return path

    def _synthesize_conclusion(self, thoughts: list[ThoughtNode]) -> str:
        """Synthesize final conclusion from thoughts."""
        if not thoughts:
            return "Unable to reach conclusion"

        # Combine insights from thought chain
        insights = [t.content for t in thoughts]
        return f"Conclusion based on {len(thoughts)} reasoning steps: " + "; ".join(insights[-2:])

    def _is_relevant(self, query1: str, query2: str) -> bool:
        """Check if two queries are related."""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

class ReasoningAgent(BaseAgent):
    def __init__(self, ...):
        super().__init__(...)
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2') # Or another suitable model

    def _is_relevant(self, query1: str, query2: str) -> bool:
        embedding1 = self.embedding_model.encode(query1)
        embedding2 = self.embedding_model.encode(query2)
        similarity = self._cosine_similarity(embedding1, embedding2)
        return similarity >= 0.7 # Adjust threshold as needed

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
