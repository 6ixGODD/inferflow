from __future__ import annotations

import abc
import asyncio
import dataclasses
import typing as t

from inferflow.asyncio.workflow import WorkflowExecutor
from inferflow.workflow import TaskMetadata

__doctitle__ = "Object-oriented Workflow (Async)"

InputT = t.TypeVar("InputT")
OutputT = t.TypeVar("OutputT")
ContextT = t.TypeVar("ContextT")


class Task(abc.ABC, t.Generic[InputT, OutputT]):
    """Abstract task with typed input and output (async version)."""

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
    async def execute(self, input: InputT) -> OutputT:
        """Execute the task."""

    def should_execute(self, _input: InputT) -> bool:
        return True


@dataclasses.dataclass
class PipelineTask(Task[bytes, t.Any]):
    """Task wrapper for InferFlow pipelines (async version)."""

    pipeline: t.Any
    name: str = ""

    async def execute(self, input: bytes) -> t.Any:
        return await self.pipeline(input)


@dataclasses.dataclass
class FunctionTask(Task[InputT, OutputT]):
    """Task wrapper for functions (async version)."""

    func: t.Callable[[InputT], t.Awaitable[OutputT] | OutputT]
    name: str = ""

    async def execute(self, input: InputT) -> OutputT:
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(input)
        return self.func(input)  # type: ignore


class TaskChain(t.Generic[InputT, OutputT]):
    """Chain of tasks with automatic type flow (async version)."""

    def __init__(self):
        self.tasks: list[Task[t.Any, t.Any]] = []

    def then(self, task: Task[t.Any, OutputT]) -> TaskChain[InputT, OutputT]:
        self.tasks.append(task)
        return self  # type: ignore

    async def execute(self, input: InputT) -> OutputT:
        current: t.Any = input
        for task in self.tasks:
            if task.should_execute(current):
                current = await task.execute(current)
        return current  # type: ignore


class ParallelTasks(t.Generic[InputT]):
    """Execute multiple tasks in parallel with same input (async version)."""

    def __init__(self, tasks: list[Task[InputT, t.Any]]):
        self.tasks = tasks

    async def execute(self, input: InputT) -> list[t.Any]:
        return await asyncio.gather(*[task.execute(input) for task in self.tasks if task.should_execute(input)])


@dataclasses.dataclass
class TypedWorkflow(WorkflowExecutor[ContextT]):
    """Typed workflow with explicit context transformation (async version)."""

    input_builder: t.Callable[[t.Any], ContextT]
    tasks: list[Task[ContextT, ContextT]]
    output_builder: t.Callable[[ContextT], t.Any] | None = None

    async def run(self, input_data: t.Any) -> t.Any:
        context = self.input_builder(input_data)

        for task in self.tasks:
            if task.should_execute(context):
                context = await task.execute(context)

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
