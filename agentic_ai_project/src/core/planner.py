"""Planner module for Agentic AI.

Provides task decomposition, planning, and scheduling capabilities.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """A single task in a plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: list[str] = field(default_factory=list)
    subtasks: list["Task"] = field(default_factory=list)
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_ready(self, completed_tasks: set[str]) -> bool:
        """Check if task is ready to execute.

        Args:
            completed_tasks: Set of completed task IDs.

        Returns:
            True if all dependencies are met.
        """
        return all(dep in completed_tasks for dep in self.dependencies)

    def mark_completed(self, result: Any = None) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result

    def mark_failed(self, error: str = "") -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.metadata["error"] = error


@dataclass
class Plan:
    """A plan consisting of tasks."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    goal: str = ""
    tasks: list[Task] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING

    def get_next_tasks(self) -> list[Task]:
        """Get tasks that are ready to execute.

        Returns:
            List of ready tasks.
        """
        completed = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        ready = []

        for task in self.tasks:
            if task.status == TaskStatus.PENDING and task.is_ready(completed):
                ready.append(task)

        # Sort by priority
        ready.sort(key=lambda t: t.priority.value, reverse=True)
        return ready

    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)

    def get_progress(self) -> float:
        """Get plan completion progress.

        Returns:
            Progress as a fraction (0.0 to 1.0).
        """
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)

    def add_task(self, task: Task) -> None:
        """Add a task to the plan."""
        self.tasks.append(task)

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None


class Planner:
    """Task planning and decomposition engine."""

    def __init__(self):
        """Initialize the planner."""
        self.plans: dict[str, Plan] = {}
        self._decomposition_strategies: dict[str, Callable] = {}

    def create_plan(self, goal: str, name: str = "") -> Plan:
        """Create a new plan for a goal.

        Args:
            goal: The goal to achieve.
            name: Optional plan name.

        Returns:
            The created plan.
        """
        plan = Plan(
            name=name or f"Plan for: {goal[:50]}",
            goal=goal,
        )
        self.plans[plan.id] = plan
        logger.info(f"Created plan: {plan.name}")
        return plan

    def decompose_goal(
        self,
        goal: str,
        strategy: str = "hierarchical",
    ) -> list[Task]:
        """Decompose a goal into tasks.

        Args:
            goal: The goal to decompose.
            strategy: Decomposition strategy.

        Returns:
            List of tasks.
        """
        if strategy in self._decomposition_strategies:
            return self._decomposition_strategies[strategy](goal)

        # Default hierarchical decomposition
        return self._hierarchical_decompose(goal)

    def register_strategy(
        self,
        name: str,
        strategy: Callable[[str], list[Task]],
    ) -> None:
        """Register a decomposition strategy.

        Args:
            name: Strategy name.
            strategy: Strategy function.
        """
        self._decomposition_strategies[name] = strategy

    def optimize_plan(self, plan: Plan) -> Plan:
        """Optimize a plan for efficiency.

        Args:
            plan: Plan to optimize.

        Returns:
            Optimized plan.
        """
        # Identify parallelizable tasks
        self._identify_parallel_tasks(plan)

        # Reorder by priority and dependencies
        self._reorder_tasks(plan)

        return plan

    def estimate_duration(self, plan: Plan) -> float:
        """Estimate total plan duration.

        Args:
            plan: Plan to estimate.

        Returns:
            Estimated duration in seconds.
        """
        if not plan.tasks:
            return 0.0

        # Build dependency graph and find critical path
        return self._calculate_critical_path(plan)

    def monitor_progress(self, plan: Plan) -> dict[str, Any]:
        """Monitor plan execution progress.

        Args:
            plan: Plan to monitor.

        Returns:
            Progress report.
        """
        total = len(plan.tasks)
        completed = sum(1 for t in plan.tasks if t.status == TaskStatus.COMPLETED)
        in_progress = sum(1 for t in plan.tasks if t.status == TaskStatus.IN_PROGRESS)
        failed = sum(1 for t in plan.tasks if t.status == TaskStatus.FAILED)
        blocked = sum(1 for t in plan.tasks if t.status == TaskStatus.BLOCKED)

        return {
            "plan_id": plan.id,
            "total_tasks": total,
            "completed": completed,
            "in_progress": in_progress,
            "failed": failed,
            "blocked": blocked,
            "progress": completed / total if total > 0 else 0.0,
            "next_tasks": [t.name for t in plan.get_next_tasks()],
        }

    def replan(self, plan: Plan, reason: str = "") -> Plan:
        """Replan after a failure or change.

        Args:
            plan: Current plan.
            reason: Reason for replanning.

        Returns:
            Updated plan.
        """
        logger.info(f"Replanning {plan.name}: {reason}")

        # Reset failed and blocked tasks
        for task in plan.tasks:
            if task.status in (TaskStatus.FAILED, TaskStatus.BLOCKED):
                task.status = TaskStatus.PENDING
                task.result = None

        # Re-optimize
        return self.optimize_plan(plan)

    def _hierarchical_decompose(self, goal: str) -> list[Task]:
        """Default hierarchical decomposition."""
        tasks = []

        # Create high-level tasks
        tasks.append(Task(
            name="Understand goal",
            description=f"Analyze and understand: {goal}",
            priority=TaskPriority.HIGH,
        ))

        tasks.append(Task(
            name="Gather resources",
            description="Identify and gather necessary resources",
            priority=TaskPriority.MEDIUM,
            dependencies=[tasks[0].id],
        ))

        tasks.append(Task(
            name="Execute main action",
            description="Perform the main task",
            priority=TaskPriority.HIGH,
            dependencies=[tasks[1].id],
        ))

        tasks.append(Task(
            name="Verify result",
            description="Verify the goal was achieved",
            priority=TaskPriority.MEDIUM,
            dependencies=[tasks[2].id],
        ))

        return tasks

    def _identify_parallel_tasks(self, plan: Plan) -> None:
        """Identify tasks that can run in parallel."""
        for i, task1 in enumerate(plan.tasks):
            for task2 in plan.tasks[i + 1:]:
                # Tasks can be parallel if neither depends on the other
                if (task1.id not in task2.dependencies and
                    task2.id not in task1.dependencies):
                    task1.metadata.setdefault("parallel_with", []).append(task2.id)

    def _reorder_tasks(self, plan: Plan) -> None:
        """Reorder tasks by priority and dependencies."""
        # Topological sort with priority consideration
        ordered = []
        remaining = plan.tasks.copy()
        completed_ids: set[str] = set()

        while remaining:
            ready = [t for t in remaining if t.is_ready(completed_ids)]
            if not ready:
                # Circular dependency or all blocked
                logger.warning("Planner: Circular dependency or all remaining tasks are blocked. Cannot reorder further.")
                break
            # Sort ready tasks by priority
            ready.sort(key=lambda t: t.priority.value, reverse=True)
            next_task = ready[0]
            ordered.append(next_task)
            remaining.remove(next_task)
            completed_ids.add(next_task.id)

        plan.tasks = ordered

    def _calculate_critical_path(self, plan: Plan) -> float:
        """Calculate critical path duration."""
        if not plan.tasks:
            return 0.0

        # Simple estimation: sum of all task durations
        # (A more sophisticated version would calculate actual critical path)
        total = sum(t.estimated_duration for t in plan.tasks)
        return total
