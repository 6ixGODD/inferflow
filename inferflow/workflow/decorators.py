from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import typing as t

from inferflow.workflow import ContextT
from inferflow.workflow import ExecutionMode
from inferflow.workflow import TaskMetadata
from inferflow.workflow import TaskNode
from inferflow.workflow import WorkflowExecutor

__doctitle__ = "Decorator-based Workflow"


@dataclass
class DecoratedTask(t.Generic[ContextT]):
    """Task created by @task decorator (sync version)."""

    func: t.Callable[[ContextT], ContextT]
    """Task function."""

    _metadata: TaskMetadata = field(default_factory=lambda: TaskMetadata(name="task"))
    """Task metadata (internal)."""

    condition: t.Callable[[ContextT], bool] | None = None
    """Execution condition."""

    @property
    def metadata(self) -> TaskMetadata:
        """Get task metadata."""
        return self._metadata

    def execute(self, context: ContextT) -> ContextT:
        """Execute the task (sync)."""
        return self.func(context)

    def should_execute(self, context: ContextT) -> bool:
        """Check if task should execute."""
        if self.condition is None:
            return True
        return self.condition(context)


@dataclass
class TaskGroup(t.Generic[ContextT]):
    """Group of tasks with execution mode (sync version)."""

    tasks: list[TaskNode[ContextT]]
    """Tasks in the group."""

    mode: ExecutionMode
    """Execution mode."""

    _metadata: TaskMetadata = field(default_factory=lambda: TaskMetadata(name="task_group"))
    """Group metadata (internal)."""

    @property
    def metadata(self) -> TaskMetadata:
        """Get task metadata."""
        return self._metadata

    def execute(self, context: ContextT) -> ContextT:
        """Execute all tasks in the group (sync)."""
        if self.mode == ExecutionMode.SEQUENTIAL:
            for task in self.tasks:
                if task.should_execute(context):
                    context = task.execute(context)
            return context

        if self.mode == ExecutionMode.PARALLEL:
            # Sync version:  sequential execution (no true parallelism)
            # For true parallel, use asyncio version
            for task in self.tasks:
                if task.should_execute(context):
                    context = task.execute(context)
            return context

        raise ValueError(f"Unknown execution mode:  {self.mode}")

    def should_execute(self, _context: ContextT) -> bool:
        """Always execute task groups."""
        return True


def task(
    name: str | None = None,
    description: str | None = None,
    condition: t.Callable[[t.Any], bool] | None = None,
    timeout: float | None = None,
    retry: int = 0,
    skip_on_error: bool = False,
) -> t.Callable[[t.Callable[[ContextT], ContextT]], DecoratedTask[ContextT]]:
    """Decorator to create a workflow task (sync version)."""

    def decorator(func: t.Callable[[ContextT], ContextT]) -> DecoratedTask[ContextT]:
        task_name = name or func.__name__
        metadata = TaskMetadata(
            name=task_name,
            description=description or func.__doc__,
            timeout=timeout,
            retry=retry,
            skip_on_error=skip_on_error,
        )

        return DecoratedTask(
            func=func,
            _metadata=metadata,
            condition=condition,
        )

    return decorator


def parallel(*tasks: TaskNode[ContextT]) -> TaskGroup[ContextT]:
    """Create a parallel task group (sync version)."""
    return TaskGroup(tasks=list(tasks), mode=ExecutionMode.PARALLEL)


def sequence(*tasks: TaskNode[ContextT]) -> TaskGroup[ContextT]:
    """Create a sequential task group (sync version)."""
    return TaskGroup(tasks=list(tasks), mode=ExecutionMode.SEQUENTIAL)


class Workflow(WorkflowExecutor[ContextT]):
    """Workflow executor for decorator-based tasks (sync version)."""

    def __init__(self, *tasks: TaskNode[ContextT]):
        self.root = sequence(*tasks) if len(tasks) > 1 else tasks[0]

    def run(self, context: ContextT) -> ContextT:
        """Execute the workflow (sync)."""
        return self.root.execute(context)

    def __enter__(self) -> t.Self:
        """Context manager entry."""
        return self

    def __exit__(self, *args: t.Any) -> None:
        """Context manager exit."""


__all__ = [
    "DecoratedTask",
    "TaskGroup",
    "task",
    "parallel",
    "sequence",
    "Workflow",
]
