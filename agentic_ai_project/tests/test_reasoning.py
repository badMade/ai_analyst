"""Tests for reasoning and core modules."""

import pytest
from src.core import (
    Memory,
    MemoryStore,
    WorkingMemory,
    Reasoner,
    ReasoningChain,
    Planner,
    Plan,
    Task,
    DecisionMaker,
    Decision,
    Executor,
    ExecutionResult,
)
from src.core.reasoning import Premise, ReasoningType
from src.core.planner import TaskStatus, TaskPriority
from src.core.decision_maker import Option, DecisionStrategy


class TestMemory:
    """Tests for Memory module."""

    def test_memory_store_basic(self):
        """Test basic memory store operations."""
        store = MemoryStore(capacity=10)
        store.store("key1", "value1")

        assert store.retrieve("key1") == "value1"
        assert store.retrieve("nonexistent") is None

    def test_memory_store_eviction(self):
        """Test LRU eviction."""
        store = MemoryStore(capacity=3)
        store.store("key1", "value1")
        store.store("key2", "value2")
        store.store("key3", "value3")
        store.store("key4", "value4")

        assert len(store) == 3
        # key1 should be evicted (least recently used)

    def test_working_memory(self):
        """Test working memory."""
        wm = WorkingMemory(capacity=3)
        wm.add("item1")
        wm.add("item2")
        wm.add("item3")
        wm.add("item4")

        assert len(wm) == 3
        items = wm.get_all()
        assert "item1" not in items
        assert "item4" in items

    def test_unified_memory(self):
        """Test unified memory system."""
        memory = Memory()
        memory.store("fact", "knowledge")
        memory.think("thought")
        memory.experience({"state": 1}, "action", "result", 0.5)

        assert memory.retrieve("fact") == "knowledge"
        assert len(memory.working) == 1

        stats = memory.get_stats()
        assert stats["long_term_items"] == 1
        assert stats["working_items"] == 1


class TestReasoning:
    """Tests for Reasoning module."""

    def test_reasoning_chain(self):
        """Test reasoning chain building."""
        chain = ReasoningChain()
        chain.add_step("Step 1", "Result 1", 0.9)
        chain.add_step("Step 2", "Result 2", 0.8)

        assert len(chain.steps) == 2
        assert chain.get_final_result() == "Result 2"
        assert chain.get_overall_confidence() == pytest.approx(0.72)

    def test_reasoner_deductive(self):
        """Test deductive reasoning."""
        reasoner = Reasoner()
        reasoner.knowledge_base["logic"] = "All A are B"

        chain = reasoner.reason("Is X a B?", reasoning_type=ReasoningType.DEDUCTIVE)
        assert len(chain.steps) > 0

    def test_reasoner_inductive(self):
        """Test inductive reasoning."""
        reasoner = Reasoner()
        observations = [
            {"color": "red", "size": "large"},
            {"color": "red", "size": "small"},
        ]

        conclusion = reasoner.induce(observations)
        assert conclusion is not None
        assert conclusion.reasoning_type == ReasoningType.INDUCTIVE

    def test_deduction(self):
        """Test premise-based deduction."""
        reasoner = Reasoner()
        premises = [
            Premise("All humans are mortal", confidence=1.0),
            Premise("Socrates is a human", confidence=1.0),
        ]

        conclusion = reasoner.deduce(premises)
        assert conclusion is not None
        assert conclusion.confidence == 1.0


class TestPlanner:
    """Tests for Planner module."""

    def test_plan_creation(self):
        """Test plan creation."""
        planner = Planner()
        plan = planner.create_plan("Build a house")

        assert plan.goal == "Build a house"
        assert plan.status == TaskStatus.PENDING

    def test_task_decomposition(self):
        """Test goal decomposition."""
        planner = Planner()
        tasks = planner.decompose_goal("Complete project")

        assert len(tasks) > 0
        assert all(isinstance(t, Task) for t in tasks)

    def test_task_dependencies(self):
        """Test task dependency handling."""
        task1 = Task(name="First", id="t1")
        task2 = Task(name="Second", id="t2", dependencies=["t1"])

        completed = set()
        assert not task2.is_ready(completed)

        completed.add("t1")
        assert task2.is_ready(completed)

    def test_plan_progress(self):
        """Test plan progress tracking."""
        planner = Planner()
        plan = planner.create_plan("Test")

        plan.add_task(Task(name="Task 1"))
        plan.add_task(Task(name="Task 2"))

        assert plan.get_progress() == 0.0

        plan.tasks[0].mark_completed()
        assert plan.get_progress() == 0.5

    def test_plan_monitoring(self):
        """Test plan monitoring."""
        planner = Planner()
        plan = planner.create_plan("Test")
        plan.add_task(Task(name="Task 1"))

        progress = planner.monitor_progress(plan)
        assert "total_tasks" in progress
        assert "progress" in progress


class TestDecisionMaker:
    """Tests for DecisionMaker module."""

    def test_decision_making(self):
        """Test basic decision making."""
        dm = DecisionMaker()
        options = [
            Option(id="1", name="Option A", utility=0.8),
            Option(id="2", name="Option B", utility=0.6),
        ]

        decision = dm.decide(options)
        assert decision.selected_option.name == "Option A"
        assert decision.confidence > 0

    def test_expected_utility(self):
        """Test expected utility decision."""
        dm = DecisionMaker()
        options = [
            Option(id="1", name="Risky", utility=1.0, probability=0.3),
            Option(id="2", name="Safe", utility=0.5, probability=0.9),
        ]

        decision = dm.decide(options, strategy=DecisionStrategy.EXPECTED_UTILITY)
        # Safe option (0.45) > Risky option (0.3)
        assert decision.selected_option.name == "Safe"

    def test_satisficing(self):
        """Test satisficing strategy."""
        dm = DecisionMaker()
        options = [
            Option(id="1", name="Good", utility=0.7),
            Option(id="2", name="Better", utility=0.9),
        ]

        decision = dm.decide(
            options,
            strategy=DecisionStrategy.SATISFICING,
            constraints={"threshold": 0.6},
        )
        # First option meeting threshold
        assert decision.selected_option.name == "Good"


class TestExecutor:
    """Tests for Executor module."""

    def test_action_registration(self):
        """Test action registration."""
        executor = Executor()

        def test_action(x, y):
            return x + y

        executor.register_action("add", test_action)
        assert "add" in executor.get_available_actions()

    def test_action_execution(self):
        """Test action execution."""
        executor = Executor()
        executor.register_action("greet", lambda name: f"Hello, {name}!")

        result = executor.execute("greet", {"name": "World"})
        assert result.status.value == "success"
        assert result.result == "Hello, World!"

    def test_execution_retry(self):
        """Test execution with retry."""
        executor = Executor(max_retries=2)
        call_count = 0

        def flaky_action():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return "success"

        executor.register_action("flaky", flaky_action)
        result = executor.execute("flaky")

        assert result.status.value == "success"
        assert result.retries > 0

    def test_execution_stats(self):
        """Test execution statistics."""
        executor = Executor()
        executor.register_action("noop", lambda: None)
        executor.execute("noop")

        stats = executor.get_execution_stats()
        assert stats["total"] == 1
        assert stats["successes"] == 1
