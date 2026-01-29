"""Memory management for Agentic AI.

Provides various memory systems for agents including working memory,
long-term memory, and episodic memory.
"""

from dataclasses import dataclass, field
from typing import Any, Iterator
from collections import OrderedDict
import time
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """A single item in memory."""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "importance": self.importance,
            "metadata": self.metadata,
        }


class MemoryStore:
    """Basic key-value memory store with LRU eviction."""

    def __init__(self, capacity: int = 1000):
        """Initialize the memory store.

        Args:
            capacity: Maximum number of items to store.
        """
        self.capacity = capacity
        self._store: OrderedDict[str, MemoryItem] = OrderedDict()

    def store(
        self,
        key: str,
        value: Any,
        importance: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a value in memory.

        Args:
            key: Unique key for the memory.
            value: Value to store.
            importance: Importance score (affects eviction).
            metadata: Optional metadata.
        """
        if key in self._store:
            # Update existing
            self._store.move_to_end(key)
            self._store[key].value = value
            self._store[key].timestamp = time.time()
        else:
            # Add new
            if len(self._store) >= self.capacity:
                self._evict()

            self._store[key] = MemoryItem(
                key=key,
                value=value,
                importance=importance,
                metadata=metadata or {},
            )

    def retrieve(self, key: str) -> Any | None:
        """Retrieve a value from memory.

        Args:
            key: Key to retrieve.

        Returns:
            Stored value or None if not found.
        """
        if key not in self._store:
            return None

        item = self._store[key]
        item.access_count += 1
        self._store.move_to_end(key)
        return item.value

    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """Search memory by query.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            Matching memory items.
        """
        results = []
        query_lower = query.lower()

        for key, item in self._store.items():
            score = self._compute_relevance(query_lower, key, item)
            if score > 0:
                results.append((score, item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:top_k]]

    def delete(self, key: str) -> bool:
        """Delete an item from memory.

        Args:
            key: Key to delete.

        Returns:
            True if deleted, False if not found.
        """
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all memory."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def _evict(self) -> None:
        """Evict least important/recently used item."""
        if not self._store:
            return

        # Score items by importance and recency
        scored = []
        now = time.time()
        for key, item in self._store.items():
            recency = 1.0 / (now - item.timestamp + 1)
            score = item.importance * 0.7 + recency * 0.3
            scored.append((score, key))

        # Remove lowest scoring item
        scored.sort(key=lambda x: x[0])
        if scored:
            del self._store[scored[0][1]]

    def _compute_relevance(self, query: str, key: str, item: MemoryItem) -> float:
        """Compute relevance score for search."""
        score = 0.0

        # Key matching
        if query in key.lower():
            score += 0.5

        # Value matching
        value_str = str(item.value).lower()
        if query in value_str:
            score += 0.3

        # Importance factor
        score *= item.importance

        return score


class WorkingMemory:
    """Short-term working memory with limited capacity."""

    def __init__(self, capacity: int = 7):
        """Initialize working memory.

        Args:
            capacity: Maximum items (default 7, based on cognitive science).
        """
        self.capacity = capacity
        self._items: list[Any] = []

    def add(self, item: Any) -> None:
        """Add item to working memory.

        Args:
            item: Item to add.
        """
        if len(self._items) >= self.capacity:
            self._items.pop(0)  # Remove oldest
        self._items.append(item)

    def get_all(self) -> list[Any]:
        """Get all items in working memory."""
        return self._items.copy()

    def get_recent(self, n: int = 3) -> list[Any]:
        """Get most recent items.

        Args:
            n: Number of items to retrieve.

        Returns:
            Most recent items.
        """
        return self._items[-n:]

    def clear(self) -> None:
        """Clear working memory."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class EpisodicMemory:
    """Memory for storing episodic experiences."""

    def __init__(self, capacity: int = 500):
        """Initialize episodic memory.

        Args:
            capacity: Maximum episodes to store.
        """
        self.capacity = capacity
        self._episodes: list[dict[str, Any]] = []

    def record_episode(
        self,
        state: Any,
        action: Any,
        result: Any,
        reward: float = 0.0,
    ) -> None:
        """Record an episode.

        Args:
            state: State before action.
            action: Action taken.
            result: Result of action.
            reward: Reward received.
        """
        episode = {
            "state": state,
            "action": action,
            "result": result,
            "reward": reward,
            "timestamp": time.time(),
        }

        if len(self._episodes) >= self.capacity:
            self._episodes.pop(0)
        self._episodes.append(episode)

    def recall_similar(self, state: Any, top_k: int = 5) -> list[dict[str, Any]]:
        """Recall episodes with similar states.

        Args:
            state: Current state.
            top_k: Number of episodes to recall.

        Returns:
            Similar episodes.
        """
        # Simple similarity based on state hash
        state_hash = self._hash_state(state)
        scored = []

        for episode in self._episodes:
            ep_hash = self._hash_state(episode["state"])
            similarity = self._compute_similarity(state_hash, ep_hash)
            scored.append((similarity, episode))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def recall_successful(self, min_reward: float = 0.5) -> list[dict[str, Any]]:
        """Recall successful episodes.

        Args:
            min_reward: Minimum reward threshold.

        Returns:
            Episodes with reward above threshold.
        """
        return [ep for ep in self._episodes if ep["reward"] >= min_reward]

    def _hash_state(self, state: Any) -> str:
        """Create a hash of the state."""
        state_str = json.dumps(state, sort_keys=True, default=str)
        return hashlib.md5(state_str.encode()).hexdigest()

    def _compute_similarity(self, hash1: str, hash2: str) -> float:
        """Compute similarity between hashes."""
        # Simple character overlap
```python
    def _compute_similarity(self, hash1: str, hash2: str) -> float:
        # Cosine similarity of state embeddings
        from sklearn.metrics.pairwise import cosine_similarity

        embedding1 = self.get_embedding(hash1) # Get embedding for hash1
        embedding2 = self.get_embedding(hash2) # Get embedding for hash2

        #Need to reshape the embeddings to a 2D array
        embedding1 = embedding1.reshape(1, -1)
        embedding2 = embedding2.reshape(1, -1)

        return cosine_similarity(embedding1, embedding2)[0][0]


class Memory:
    """Unified memory system combining different memory types."""

    def __init__(
        self,
        long_term_capacity: int = 10000,
        working_capacity: int = 7,
        episodic_capacity: int = 500,
    ):
        """Initialize the unified memory system.

        Args:
            long_term_capacity: Capacity for long-term store.
            working_capacity: Capacity for working memory.
            episodic_capacity: Capacity for episodic memory.
        """
        self.long_term = MemoryStore(capacity=long_term_capacity)
        self.working = WorkingMemory(capacity=working_capacity)
        self.episodic = EpisodicMemory(capacity=episodic_capacity)

    def store(self, key: str, value: Any, **kwargs) -> None:
        """Store in long-term memory."""
        self.long_term.store(key, value, **kwargs)

    def retrieve(self, key: str) -> Any | None:
        """Retrieve from long-term memory."""
        return self.long_term.retrieve(key)

    def think(self, item: Any) -> None:
        """Add to working memory."""
        self.working.add(item)

    def experience(self, state: Any, action: Any, result: Any, reward: float = 0.0) -> None:
        """Record an experience."""
        self.episodic.record_episode(state, action, result, reward)

    def recall(self, query: str | None = None, state: Any = None) -> list[Any]:
        """Recall relevant memories.

        Args:
            query: Text query for long-term memory.
            state: State for episodic memory.

        Returns:
            Combined relevant memories.
        """
        results = []

        if query:
            results.extend(self.long_term.search(query))

        if state:
            results.extend(self.episodic.recall_similar(state))

        results.extend(self.working.get_all())

        return results

    def clear_all(self) -> None:
        """Clear all memory systems."""
        self.long_term.clear()
        self.working.clear()
        self.episodic._episodes.clear()

    def get_stats(self) -> dict[str, int]:
        """Get memory statistics."""
        return {
            "long_term_items": len(self.long_term),
            "working_items": len(self.working),
            "episodic_items": len(self.episodic._episodes),
        }
