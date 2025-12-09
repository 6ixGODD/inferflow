# inferflow/workflow/tasks.py

from __future__ import annotations

import abc
import dataclasses
import typing as t

from inferflow.workflow import TaskMetadata
from inferflow.workflow import WorkflowExecutor

__doctitle__ = "Object-oriented Workflow"

InputT = t.TypeVar("InputT")
OutputT = t.TypeVar("OutputT")
ContextT = t.TypeVar("ContextT")


class Task(abc.ABC, t.Generic[InputT, OutputT]):
    """Abstract task with typed input and output (sync version)."""

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        timeout: float | None = None,
        retry: int = 0,
        skip_on_error: bool = False,
    ):
        self.metadata = TaskMetadata(
            name=name or self.__class__.__name__,
            description=description or self.__class__.__doc__,
            timeout=timeout,
            retry=retry,
            skip_on_error=skip_on_error,
        )

    @abc.abstractmethod
    def execute(self, input: InputT) -> OutputT:
        """Execute the task (sync)."""

    def should_execute(self, _input: InputT) -> bool:
        """Check if task should execute."""
        return True


@dataclasses.dataclass
class PipelineTask(Task[bytes, t.Any]):
    """Task wrapper for InferFlow pipelines (sync version)."""

    pipeline: t.Any
    """InferFlow pipeline instance."""

    name: str = ""
    """Task name."""

    def execute(self, input: bytes) -> t.Any:
        """Execute pipeline inference (sync)."""
        return self.pipeline(input)


@dataclasses.dataclass
class FunctionTask(Task[InputT, OutputT]):
    """Task wrapper for functions (sync version)."""

    func: t.Callable[[InputT], OutputT]
    """Task function."""

    name: str = ""
    """Task name."""

    def execute(self, input: InputT) -> OutputT:
        """Execute function (sync)."""
        return self.func(input)


class TaskChain(t.Generic[InputT, OutputT]):
    """Chain of tasks with automatic type flow (sync version)."""

    def __init__(self):
        self.tasks: list[Task[t.Any, t.Any]] = []

    def then(self, task: Task[t.Any, OutputT]) -> TaskChain[InputT, OutputT]:
        """Add a task to the chain."""
        self.tasks.append(task)
        return self  # type: ignore

    def execute(self, input: InputT) -> OutputT:
        """Execute all tasks in sequence (sync)."""
        current: t.Any = input

        for task in self.tasks:
            if task.should_execute(current):
                current = task.execute(current)

        return current  # type: ignore


class ParallelTasks(t.Generic[InputT]):
    """Execute multiple tasks in parallel with same input (sync version).

    Note: In sync version, tasks are executed sequentially.
    Use asyncio version for true parallelism.
    """

    def __init__(self, tasks: list[Task[InputT, t.Any]]):
        self.tasks = tasks

    def execute(self, input: InputT) -> list[t.Any]:
        """Execute all tasks (sync - sequential)."""
        results = []
        for task in self.tasks:
            if task.should_execute(input):
                results.append(task.execute(input))
        return results


@dataclasses.dataclass
class TypedWorkflow(WorkflowExecutor[ContextT]):
    """Typed workflow with explicit context transformation (sync version)."""

    input_builder: t.Callable[[t.Any], ContextT]
    """Build initial context from input."""

    tasks: list[Task[ContextT, ContextT]]
    """Tasks to execute."""

    output_builder: t.Callable[[ContextT], t.Any] | None = None
    """Extract output from final context."""

    def run(self, input_data: t.Any) -> t.Any:
        """Execute workflow (sync)."""
        context = self.input_builder(input_data)

        for task in self.tasks:
            if task.should_execute(context):
                context = task.execute(context)

        if self.output_builder:
            return self.output_builder(context)
        return context


__all__ = [
    "Task",
    "PipelineTask",
    "FunctionTask",
    "TaskChain",
    "ParallelTasks",
    "TypedWorkflow",
]
