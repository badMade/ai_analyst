"""Executor module for Agentic AI.

Provides action execution, monitoring, and error handling.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine
from enum import Enum
import asyncio
import logging
import time
import traceback

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of an execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of an action execution."""
    status: ExecutionStatus
    result: Any = None
    error: str | None = None
    duration: float = 0.0
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context for action execution."""
    action_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0


class Executor:
    """Action execution engine."""

    def __init__(
        self,
        default_timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize the executor.

        Args:
            default_timeout: Default execution timeout in seconds.
            max_retries: Default maximum retry attempts.
        """
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self._actions: dict[str, Callable] = {}
        self._hooks_before: list[Callable] = []
        self._hooks_after: list[Callable] = []
        self._execution_history: list[ExecutionResult] = []

    def register_action(
        self,
        name: str,
        action: Callable,
        description: str = "",
    ) -> None:
        """Register an executable action.

        Args:
            name: Action name.
            action: Action function.
            description: Action description.
        """
        self._actions[name] = action
        logger.debug(f"Registered action: {name}")

    def execute(
        self,
        action_name: str,
        parameters: dict[str, Any] | None = None,
        context: ExecutionContext | None = None,
    ) -> ExecutionResult:
        """Execute an action synchronously.

        Args:
            action_name: Name of action to execute.
            parameters: Action parameters.
            context: Execution context.

        Returns:
            Execution result.
        """
        context = context or ExecutionContext(
            action_name=action_name,
            parameters=parameters or {},
            timeout=self.default_timeout,
            max_retries=self.max_retries,
        )

        # Run before hooks
        self._run_hooks(self._hooks_before, context)

        result = self._execute_with_retry(context)

        # Run after hooks
        self._run_hooks(self._hooks_after, context, result)

        self._execution_history.append(result)
        return result

    async def execute_async(
        self,
        action_name: str,
        parameters: dict[str, Any] | None = None,
        context: ExecutionContext | None = None,
    ) -> ExecutionResult:
        """Execute an action asynchronously.

        Args:
            action_name: Name of action to execute.
            parameters: Action parameters.
            context: Execution context.

        Returns:
            Execution result.
        """
        context = context or ExecutionContext(
            action_name=action_name,
            parameters=parameters or {},
            timeout=self.default_timeout,
            max_retries=self.max_retries,
        )

        # Run before hooks
        self._run_hooks(self._hooks_before, context)

        result = await self._execute_with_retry_async(context)

        # Run after hooks
        self._run_hooks(self._hooks_after, context, result)

        self._execution_history.append(result)
        return result

    def execute_batch(
        self,
        actions: list[tuple[str, dict[str, Any]]],
        parallel: bool = False,
    ) -> list[ExecutionResult]:
        """Execute multiple actions.

        Args:
            actions: List of (action_name, parameters) tuples.
            parallel: Whether to execute in parallel.

        Returns:
            List of execution results.
        """
        if parallel:
            return asyncio.run(self._execute_batch_async(actions))

        results = []
        for action_name, params in actions:
            result = self.execute(action_name, params)
            results.append(result)
        return results

    def add_before_hook(self, hook: Callable[[ExecutionContext], None]) -> None:
        """Add a hook to run before execution.

        Args:
            hook: Hook function.
        """
        self._hooks_before.append(hook)

    def add_after_hook(
        self,
        hook: Callable[[ExecutionContext, ExecutionResult], None],
    ) -> None:
        """Add a hook to run after execution.

        Args:
            hook: Hook function.
        """
        self._hooks_after.append(hook)

    def get_available_actions(self) -> list[str]:
        """Get list of registered action names."""
        return list(self._actions.keys())

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics.

        Returns:
            Statistics dictionary.
        """
        total = len(self._execution_history)
        if total == 0:
            return {"total": 0}

        successes = sum(
            1 for r in self._execution_history
            if r.status == ExecutionStatus.SUCCESS
        )
        failures = sum(
            1 for r in self._execution_history
            if r.status == ExecutionStatus.FAILED
        )
        timeouts = sum(
            1 for r in self._execution_history
            if r.status == ExecutionStatus.TIMEOUT
        )
        total_duration = sum(r.duration for r in self._execution_history)
        avg_duration = total_duration / total

        return {
            "total": total,
            "successes": successes,
            "failures": failures,
            "timeouts": timeouts,
            "success_rate": successes / total,
            "avg_duration": avg_duration,
            "total_duration": total_duration,
        }

    def _execute_with_retry(self, context: ExecutionContext) -> ExecutionResult:
        """Execute with retry logic."""
        action = self._actions.get(context.action_name)

        if action is None:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Unknown action: {context.action_name}",
            )

        retries = 0
        last_error = None

        while retries <= context.max_retries:
            start_time = time.time()

            try:
                # Execute with timeout
                result = self._execute_with_timeout(
                    action,
                    context.parameters,
                    context.timeout,
                )

                duration = time.time() - start_time
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    result=result,
                    duration=duration,
                    retries=retries,
                )

            except TimeoutError:
                last_error = "Execution timed out"
                logger.warning(
                    f"Action {context.action_name} timed out (attempt {retries + 1})"
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Action {context.action_name} failed: {e} (attempt {retries + 1})"
                )

            retries += 1
            if retries <= context.max_retries:
                time.sleep(context.retry_delay * retries)

        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            error=last_error,
            duration=time.time() - start_time,
            retries=retries - 1,
        )

    async def _execute_with_retry_async(
        self,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Execute asynchronously with retry logic."""
        action = self._actions.get(context.action_name)

        if action is None:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Unknown action: {context.action_name}",
            )

        retries = 0
        last_error = None
        start_time = time.time()

        while retries <= context.max_retries:
            try:
                if asyncio.iscoroutinefunction(action):
                    result = await asyncio.wait_for(
                        action(**context.parameters),
                        timeout=context.timeout,
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(action, **context.parameters),
                        timeout=context.timeout,
                    )

                duration = time.time() - start_time
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    result=result,
                    duration=duration,
                    retries=retries,
                )

            except asyncio.TimeoutError:
                last_error = "Execution timed out"
                logger.warning(
                    f"Action {context.action_name} timed out (attempt {retries + 1})"
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Action {context.action_name} failed: {e} (attempt {retries + 1})"
                )

            retries += 1
            if retries <= context.max_retries:
                await asyncio.sleep(context.retry_delay * retries)

        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            error=last_error,
            duration=time.time() - start_time,
            retries=retries - 1,
        )

    async def _execute_batch_async(
        self,
        actions: list[tuple[str, dict[str, Any]]],
    ) -> list[ExecutionResult]:
        """Execute batch of actions asynchronously."""
        tasks = [
            self.execute_async(name, params)
            for name, params in actions
        ]
        return await asyncio.gather(*tasks)

    def _execute_with_timeout(
        self,
        action: Callable,
        parameters: dict[str, Any],
        timeout: float,
    ) -> Any:
        """Execute action with timeout (synchronous)."""
        # Simple synchronous execution - timeout handled differently
        return action(**parameters)

    def _run_hooks(
        self,
        hooks: list[Callable],
        context: ExecutionContext,
        result: ExecutionResult | None = None,
    ) -> None:
        """Run execution hooks."""
        for hook in hooks:
            try:
                if result is not None:
                    hook(context, result)
                else:
                    hook(context)
            except Exception as e:
                logger.error(f"Hook error: {e}")
