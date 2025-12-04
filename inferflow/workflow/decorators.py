from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
import typing as t

from inferflow.workflow import ContextT
from inferflow.workflow import ExecutionMode
from inferflow.workflow import TaskMetadata
from inferflow.workflow import TaskNode
from inferflow.workflow import WorkflowExecutor


@dataclass
class DecoratedTask(t.Generic[ContextT]):
    """Task created by @task decorator."""

    func: t.Callable[[ContextT], t.Awaitable[ContextT]]
    """Task function."""

    _metadata: TaskMetadata = field(default_factory=lambda: TaskMetadata(name="task"))
    """Task metadata (internal)."""

    condition: t.Callable[[ContextT], bool] | None = None
    """Execution condition."""

    @property
    def metadata(self) -> TaskMetadata:
        """Get task metadata."""
        return self._metadata

    async def execute(self, context: ContextT) -> ContextT:
        """Execute the task."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(context)
        return self.func(context)  # type: ignore

    def should_execute(self, context: ContextT) -> bool:
        """Check if task should execute."""
        if self.condition is None:
            return True
        return self.condition(context)


@dataclass
class TaskGroup(t.Generic[ContextT]):
    """Group of tasks with execution mode."""

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

    async def execute(self, context: ContextT) -> ContextT:
        """Execute all tasks in the group."""
        if self.mode == ExecutionMode.SEQUENTIAL:
            for task in self.tasks:
                if task.should_execute(context):
                    context = await task.execute(context)
            return context

        if self.mode == ExecutionMode.PARALLEL:
            # Execute all tasks with current context
            results = await asyncio.gather(*[
                task.execute(context) for task in self.tasks if task.should_execute(context)
            ])

            # Merge results (last write wins)
            for result in results:
                context = result

            return context

        raise ValueError(f"Unknown execution mode: {self.mode}")

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
) -> t.Callable[
    [t.Callable[[ContextT], t.Awaitable[ContextT]]],
    DecoratedTask[ContextT],
]:
    """Decorator to create a workflow task.

    Args:
        name: Task name (defaults to function name).
        description: Task description.
        condition: Condition function to check if task should execute.
        timeout: Execution timeout in seconds.
        retry: Number of retries on failure.
        skip_on_error: Continue workflow if task fails.

    Returns:
        Decorated task.

    Example:
        ```python
        @task(
            name="process_image",
            condition=lambda ctx: ctx.image is not None,
        )
        async def process(ctx: MyContext) -> MyContext:
            ctx.result = await pipeline(ctx.image)
            return ctx
        ```
    """

    def decorator(func: t.Callable[[ContextT], t.Awaitable[ContextT]]) -> DecoratedTask[ContextT]:
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
            _metadata=metadata,  # 使用 _metadata
            condition=condition,
        )

    return decorator


def parallel(*tasks: TaskNode[ContextT]) -> TaskGroup[ContextT]:
    """Create a parallel task group.

    Args:
        *tasks: Tasks to execute in parallel.

    Returns:
        Task group with parallel execution.

    Example:
        ```python
        parallel(
            classify_color,
            detect_cracks,
            analyze_texture,
        )
        ```
    """
    return TaskGroup(
        tasks=list(tasks),
        mode=ExecutionMode.PARALLEL,
    )


def sequence(*tasks: TaskNode[ContextT]) -> TaskGroup[ContextT]:
    """Create a sequential task group.

    Args:
        *tasks: Tasks to execute sequentially.

    Returns:
        Task group with sequential execution.

    Example:
        ```python
        sequence(
            load_image,
            preprocess,
            infer,
            postprocess,
        )
        ```
    """
    return TaskGroup(
        tasks=list(tasks),
        mode=ExecutionMode.SEQUENTIAL,
    )


class Workflow(WorkflowExecutor[ContextT]):
    """Workflow executor for decorator-based tasks.

    Args:
        *tasks: Root tasks or task groups.

    Example:
        ```python
        workflow = Workflow[MyContext](
            load_data,
            parallel(
                process_a,
                process_b,
            ),
            finalize,
        )

        ctx = MyContext(input="data")
        result = await workflow.run(ctx)
        ```
    """

    def __init__(self, *tasks: TaskNode[ContextT]):
        self.root = sequence(*tasks) if len(tasks) > 1 else tasks[0]

    async def run(self, context: ContextT) -> ContextT:
        """Execute the workflow.

        Args:
            context: Initial context.

        Returns:
            Final context after execution.
        """
        return await self.root.execute(context)

    async def __aenter__(self) -> t.Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: t.Any) -> None:
        """Async context manager exit."""
