from __future__ import annotations

import asyncio
import dataclasses
import typing as t

from inferflow.asyncio.workflow import ContextT
from inferflow.asyncio.workflow import TaskNode
from inferflow.asyncio.workflow import WorkflowExecutor
from inferflow.workflow import ExecutionMode
from inferflow.workflow import TaskMetadata

__doctitle__ = "Decorator-based Workflow (Async)"


@dataclasses.dataclass
class DecoratedTask(t.Generic[ContextT]):
    """Task created by @task decorator (async version)."""

    func: t.Callable[[ContextT], t.Awaitable[ContextT]]
    _metadata: TaskMetadata = dataclasses.field(default_factory=lambda: TaskMetadata(name="task"))
    condition: t.Callable[[ContextT], bool] | None = None

    @property
    def metadata(self) -> TaskMetadata:
        return self._metadata

    async def execute(self, context: ContextT) -> ContextT:
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(context)
        return self.func(context)  # type: ignore

    def should_execute(self, context: ContextT) -> bool:
        if self.condition is None:
            return True
        return self.condition(context)


@dataclasses.dataclass
class TaskGroup(t.Generic[ContextT]):
    """Group of tasks with execution mode (async version)."""

    tasks: list[TaskNode[ContextT]]
    mode: ExecutionMode
    _metadata: TaskMetadata = dataclasses.field(default_factory=lambda: TaskMetadata(name="task_group"))

    @property
    def metadata(self) -> TaskMetadata:
        return self._metadata

    async def execute(self, context: ContextT) -> ContextT:
        if self.mode == ExecutionMode.SEQUENTIAL:
            for task in self.tasks:
                if task.should_execute(context):
                    context = await task.execute(context)
            return context

        if self.mode == ExecutionMode.PARALLEL:
            results = await asyncio.gather(*[
                task.execute(context) for task in self.tasks if task.should_execute(context)
            ])
            for result in results:
                context = result
            return context

        raise ValueError(f"Unknown execution mode:  {self.mode}")

    def should_execute(self, _context: ContextT) -> bool:
        return True


def task(
    name: str | None = None,
    description: str | None = None,
    condition: t.Callable[[t.Any], bool] | None = None,
    timeout: float | None = None,
    retry: int = 0,
    skip_on_error: bool = False,
) -> t.Callable[[t.Callable[[ContextT], t.Awaitable[ContextT]]], DecoratedTask[ContextT]]:
    """Decorator to create a workflow task (async version)."""

    def decorator(func: t.Callable[[ContextT], t.Awaitable[ContextT]]) -> DecoratedTask[ContextT]:
        task_name = name or func.__name__
        metadata = TaskMetadata(
            name=task_name,
            description=description or func.__doc__,
            timeout=timeout,
            retry=retry,
            skip_on_error=skip_on_error,
        )
        return DecoratedTask(func=func, _metadata=metadata, condition=condition)

    return decorator


def parallel(*tasks: TaskNode[ContextT]) -> TaskGroup[ContextT]:
    """Create a parallel task group (async version)."""
    return TaskGroup(tasks=list(tasks), mode=ExecutionMode.PARALLEL)


def sequence(*tasks: TaskNode[ContextT]) -> TaskGroup[ContextT]:
    """Create a sequential task group (async version)."""
    return TaskGroup(tasks=list(tasks), mode=ExecutionMode.SEQUENTIAL)


class Workflow(WorkflowExecutor[ContextT]):
    """Workflow executor for decorator-based tasks (async version)."""

    def __init__(self, *tasks: TaskNode[ContextT]):
        self.root = sequence(*tasks) if len(tasks) > 1 else tasks[0]

    async def run(self, context: ContextT) -> ContextT:
        return await self.root.execute(context)

    async def __aenter__(self) -> t.Self:
        return self

    async def __aexit__(self, *args: t.Any) -> None:
        pass


__all__ = ["DecoratedTask", "TaskGroup", "task", "parallel", "sequence", "Workflow"]
