from __future__ import annotations

import asyncio
import time

import pytest
import torch


@pytest.mark.benchmark
class TestAsyncVsSyncPerformance:
    """Compare async vs sync pipeline performance."""

    @pytest.fixture
    def dummy_input(self):
        """Dummy input tensor."""
        return torch.randn(3, 224, 224)

    def test_sync_sequential(self, dummy_input):
        """Benchmark sync pipeline (sequential)."""
        from inferflow.pipeline.classification.torch import ClassificationPipeline
        from tests.conftest import MockClassificationRuntime

        runtime = MockClassificationRuntime(processing_time_ms=10.0)
        pipeline = ClassificationPipeline(runtime=runtime)

        num_requests = 50

        with pipeline.serve():
            start = time.perf_counter()

            for _ in range(num_requests):
                pipeline(dummy_input)

            elapsed = time.perf_counter() - start

        print(f"\n{'=' * 80}")
        print(f"📊 Sync Sequential ({num_requests} requests)")
        print(f"{'=' * 80}")
        print(f"  Total time:  {elapsed * 1000:8.2f} ms")
        print(f"  Throughput:  {num_requests / elapsed:8.1f} req/s")

    @pytest.mark.asyncio
    async def test_async_sequential(self, dummy_input):
        """Benchmark async pipeline (sequential)."""
        from inferflow.asyncio.pipeline.classification.torch import ClassificationPipeline
        from tests.conftest import AsyncMockClassificationRuntime

        runtime = AsyncMockClassificationRuntime(processing_time_ms=10.0)
        pipeline = ClassificationPipeline(runtime=runtime)

        num_requests = 50

        async with pipeline.serve():
            start = time.perf_counter()

            for _ in range(num_requests):
                await pipeline(dummy_input)

            elapsed = time.perf_counter() - start

        print(f"\n{'=' * 80}")
        print(f"📊 Async Sequential ({num_requests} requests)")
        print(f"{'=' * 80}")
        print(f"  Total time:  {elapsed * 1000:8.2f} ms")
        print(f"  Throughput:   {num_requests / elapsed: 8.1f} req/s")

    @pytest.mark.asyncio
    async def test_async_concurrent(self, dummy_input):
        """Benchmark async pipeline (concurrent)."""
        from inferflow.asyncio.pipeline.classification.torch import ClassificationPipeline
        from tests.conftest import AsyncMockClassificationRuntime

        runtime = AsyncMockClassificationRuntime(processing_time_ms=10.0)
        pipeline = ClassificationPipeline(runtime=runtime)

        num_requests = 50

        async with pipeline.serve():
            start = time.perf_counter()

            tasks = [pipeline(dummy_input) for _ in range(num_requests)]
            await asyncio.gather(*tasks)

            elapsed = time.perf_counter() - start

        print(f"\n{'=' * 80}")
        print(f"📊 Async Concurrent ({num_requests} requests)")
        print(f"{'=' * 80}")
        print(f"  Total time:  {elapsed * 1000:8.2f} ms")
        print(f"  Throughput:  {num_requests / elapsed:8.1f} req/s")
        print(f"  Speedup:     {(num_requests * 10.0 / 1000) / elapsed:8.2f}x")


__all__ = ["TestAsyncVsSyncPerformance"]
