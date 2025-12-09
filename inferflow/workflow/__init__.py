from __future__ import annotations

import abc
import dataclasses
import enum
import importlib
import typing as t

__doctitle__ = "Workflow"

T_co = t.TypeVar("T_co", covariant=True)
ContextT = t.TypeVar("ContextT")


class ExecutionMode(str, enum.Enum):
    """Task execution mode."""

    SEQUENTIAL = "sequential"
    """Execute tasks one by one."""

    PARALLEL = "parallel"
    """Execute tasks concurrently."""


@dataclasses.dataclass
class TaskMetadata:
    """Metadata for a workflow task."""

    name: str
    """Task name."""

    description: str | None = None
    """Task description."""

    timeout: float | None = None
    """Execution timeout in seconds."""

    retry: int = 0
    """Number of retries on failure."""

    skip_on_error: bool = False
    """Continue workflow if task fails."""


class TaskNode(t.Protocol[ContextT]):
    """Protocol for a workflow task node (sync version)."""

    @property
    def metadata(self) -> TaskMetadata:
        """Get task metadata."""

    def execute(self, context: ContextT) -> ContextT:
        """Execute the task.

        Args:
            context:  Workflow context.

        Returns:
            Updated context.
        """

    def should_execute(self, context: ContextT) -> bool:
        """Check if task should execute.

        Args:
            context:  Current context.

        Returns:
            True if task should execute.
        """


class WorkflowExecutor(abc.ABC, t.Generic[ContextT]):
    """Abstract workflow executor (sync version)."""

    @abc.abstractmethod
    def run(self, context: ContextT) -> ContextT:
        """Execute the workflow.

        Args:
            context: Initial context.

        Returns:
            Final context after execution.
        """


__all__ = [
    "ExecutionMode",
    "TaskMetadata",
    "TaskNode",
    "WorkflowExecutor",
    "decorators",
    "tasks",
]


def __getattr__(name: str) -> t.Any:
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
