from __future__ import annotations

import asyncio
import time

import pytest
import torch


def timer_async(func, *args, warmup: int = 2, repeat: int = 10, **kwargs):
    """Measure async function execution time."""

    async def run_benchmark():
        # Warmup
        for _ in range(warmup):
            await func(*args, **kwargs)

        # Timed execution
        start = time.perf_counter()
        for _ in range(repeat):
            await func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        return elapsed / repeat

    return asyncio.run(run_benchmark())


@pytest.mark.benchmark
class TestBatchStrategyPerformance:
    """Benchmark batch strategy performance."""

    @pytest.mark.asyncio
    async def test_single_request_latency(self):
        """Measure single request latency."""
        from inferflow.asyncio.batch.dynamic import DynamicBatchStrategy
        from tests.conftest import AsyncMockRuntime

        runtime = AsyncMockRuntime(processing_time_ms=1.0)
        await runtime.load()
        strategy = DynamicBatchStrategy(max_batch_size=8, max_wait_ms=50.0)
        await strategy.start(runtime)

        num_requests = 100

        start = time.perf_counter()
        for i in range(num_requests):
            await strategy.submit(torch.tensor([float(i)]))
        elapsed = time.perf_counter() - start

        await strategy.stop()

        print(f"\n{'=' * 80}")
        print(f"📊 Single Request Latency (sequential, {num_requests} requests)")
        print(f"{'=' * 80}")
        print(f"  Total time:   {elapsed * 1000:8.2f} ms")
        print(f"  Throughput:  {num_requests / elapsed:8.1f} req/s")
        print(f"  Avg latency: {elapsed * 1000 / num_requests:8.2f} ms/req")
        await runtime.unload()

    @pytest.mark.asyncio
    async def test_concurrent_throughput(self):
        """Measure concurrent request throughput."""
        from inferflow.asyncio.batch.dynamic import DynamicBatchStrategy
        from tests.conftest import AsyncMockRuntime

        runtime = AsyncMockRuntime(processing_time_ms=5.0)
        await runtime.load()

        strategy = DynamicBatchStrategy(
            max_batch_size=32,
            max_wait_ms=50.0,
            queue_size=1000,
        )
        await strategy.start(runtime)

        num_requests = 200

        start = time.perf_counter()

        tasks = [strategy.submit(torch.tensor([float(i)])) for i in range(num_requests)]
        await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start

        metrics = strategy.get_metrics()
        await strategy.stop()

        print(f"\n{'=' * 80}")
        print(f"📊 Concurrent Throughput ({num_requests} concurrent requests)")
        print(f"{'=' * 80}")
        print(f"  Total time:    {elapsed * 1000:8.2f} ms")
        print(f"  Throughput:   {num_requests / elapsed:8.1f} req/s")
        print(f"  Total batches:{metrics.total_batches:8d}")
        print(f"  Avg batch:     {metrics.avg_batch_size:8.2f}")
        await runtime.unload()

    @pytest.mark.asyncio
    async def test_sustained_load(self):
        """Measure performance under sustained load."""
        from inferflow.asyncio.batch.dynamic import DynamicBatchStrategy
        from tests.conftest import AsyncMockRuntime

        runtime = AsyncMockRuntime(processing_time_ms=2.0)
        await runtime.load()
        duration_seconds = 3.0

        strategy = DynamicBatchStrategy(
            max_batch_size=32,
            max_wait_ms=50.0,
            queue_size=2000,
        )
        await strategy.start(runtime)

        start_time = time.time()
        request_count = 0

        async def submit_continuously():
            nonlocal request_count
            while time.time() - start_time < duration_seconds:
                await strategy.submit(torch.tensor([1.0]))
                request_count += 1

        # Multiple concurrent submitters
        await asyncio.gather(*[submit_continuously() for _ in range(10)])

        metrics = strategy.get_metrics()
        await strategy.stop()

        print(f"\n{'=' * 80}")
        print(f"📊 Sustained Load ({duration_seconds}s, 10 concurrent submitters)")
        print(f"{'=' * 80}")
        print(f"  Total requests: {request_count:8d}")
        print(f"  Throughput:     {request_count / duration_seconds:8.1f} req/s")
        print(f"  Avg batch:       {metrics.avg_batch_size:8.2f}")
        await runtime.unload()


__all__ = ["TestBatchStrategyPerformance"]
