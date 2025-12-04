from __future__ import annotations

import abc
import asyncio
import dataclasses
import typing as t

from inferflow.workflow import TaskMetadata
from inferflow.workflow import WorkflowExecutor

InputT = t.TypeVar("InputT")
OutputT = t.TypeVar("OutputT")
ContextT = t.TypeVar("ContextT")


class Task(abc.ABC, t.Generic[InputT, OutputT]):
    """Abstract task with typed input and output.

    Type Parameters:
        InputT: Input data type.
        OutputT: Output data type.

    Example:
        ```python
        class MyTask(Task[ImageInput, ClassificationOutput]):
            async def execute(
                self, input: ImageInput
            ) -> ClassificationOutput:
                return await self.pipeline(input.image)
        ```
    """

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
        """Execute the task.

        Args:
            input: Task input.

        Returns:
            Task output.
        """

    def should_execute(self, _input: InputT) -> bool:
        """Check if task should execute.

        Returns:
            True if task should execute (default: always True).
        """
        return True


@dataclasses.dataclass
class PipelineTask(Task[bytes, t.Any]):
    """Task wrapper for InferFlow pipelines."""

    pipeline: t.Any
    """InferFlow pipeline instance."""

    name: str = ""
    """Task name."""

    async def execute(self, input: bytes) -> t.Any:
        """Execute pipeline inference."""
        return await self.pipeline(input)


@dataclasses.dataclass
class FunctionTask(Task[InputT, OutputT]):
    """Task wrapper for functions."""

    func: t.Callable[[InputT], t.Awaitable[OutputT] | OutputT]
    """Task function."""

    name: str = ""
    """Task name."""

    async def execute(self, input: InputT) -> OutputT:
        """Execute function."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(input)
        return self.func(input)  # type: ignore


class TaskChain(t.Generic[InputT, OutputT]):
    """Chain of tasks with automatic type flow.

    Example:
        ```python
        chain = (
            TaskChain[bytes, SegmentationOutput]()
            .then(SegmentTask())
            .then(CropTask())
            .then(ClassifyTask())
        )

        result = await chain.execute(image_bytes)
        ```
    """

    def __init__(self):
        self.tasks: list[Task[t.Any, t.Any]] = []

    def then(self, task: Task[t.Any, OutputT]) -> TaskChain[InputT, OutputT]:
        """Add a task to the chain.

        Args:
            task: Task to add.

        Returns:
            Self for chaining.
        """
        self.tasks.append(task)
        return self  # type: ignore

    async def execute(self, input: InputT) -> OutputT:
        """Execute all tasks in sequence.

        Args:
            input: Initial input.

        Returns:
            Final output.
        """
        current: t.Any = input

        for task in self.tasks:
            if task.should_execute(current):
                current = await task.execute(current)

        return current  # type: ignore


class ParallelTasks(t.Generic[InputT]):
    """Execute multiple tasks in parallel with same input.

    Example:
        ```python
        parallel = ParallelTasks[bytes]([
            ClassifyColorTask(),
            DetectCracksTask(),
            AnalyzeTextureTask(),
        ])

        results = await parallel.execute(image_bytes)
        ```
    """

    def __init__(self, tasks: list[Task[InputT, t.Any]]):
        self.tasks = tasks

    async def execute(self, input: InputT) -> list[t.Any]:
        """Execute all tasks in parallel.

        Args:
            input: Input for all tasks.

        Returns:
            List of outputs from all tasks.
        """
        return await asyncio.gather(*[task.execute(input) for task in self.tasks if task.should_execute(input)])


@dataclasses.dataclass
class TypedWorkflow(WorkflowExecutor[ContextT]):
    """Typed workflow with explicit context transformation.

    Example:
        ```python
        class MyContext:
            image: bytes
            result: ClassificationOutput | None = None


        workflow = TypedWorkflow[MyContext](
            input_builder=lambda image: MyContext(image=image),
            tasks=[segment_task, classify_task],
            output_builder=lambda ctx: ctx.result,
        )

        result = await workflow.run(image_bytes)
        ```
    """

    input_builder: t.Callable[[t.Any], ContextT]
    """Build initial context from input."""

    tasks: list[Task[ContextT, ContextT]]
    """Tasks to execute."""

    output_builder: t.Callable[[ContextT], t.Any] | None = None
    """Extract output from final context."""

    async def run(self, input_data: t.Any) -> t.Any:
        """Execute workflow.

        Args:
            input_data: Input data.

        Returns:
            Workflow output.
        """
        # Build context
        context = self.input_builder(input_data)

        # Execute tasks
        for task in self.tasks:
            if task.should_execute(context):
                context = await task.execute(context)

        # Extract output
        if self.output_builder:
            return self.output_builder(context)
        return context
