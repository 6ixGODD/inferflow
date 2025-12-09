from __future__ import annotations

import abc
import importlib
import typing as t

from inferflow.workflow import ExecutionMode
from inferflow.workflow import TaskMetadata

ContextT = t.TypeVar("ContextT")


class TaskNode(t.Protocol[ContextT]):
    """Protocol for a workflow task node (async version)."""

    @property
    def metadata(self) -> TaskMetadata:
        """Get task metadata."""

    async def execute(self, context: ContextT) -> ContextT:
        """Execute the task."""

    def should_execute(self, context: ContextT) -> bool:
        """Check if task should execute."""


class WorkflowExecutor(abc.ABC, t.Generic[ContextT]):
    """Abstract workflow executor (async version)."""

    @abc.abstractmethod
    async def run(self, context: ContextT) -> ContextT:
        """Execute the workflow."""


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
