from __future__ import annotations

import asyncio

import pytest
import torch


@pytest.mark.unit
class TestDynamicBatchStrategy:
    """Unit tests for DynamicBatchStrategy."""

    @pytest.mark.asyncio
    async def test_basic_submission(self, batch_strategy):
        """Test basic request submission."""
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = await batch_strategy.submit(input_tensor)

        # Mock runtime doubles the input
        assert torch.allclose(result, input_tensor * 2)

    @pytest.mark.asyncio
    async def test_concurrent_submissions(self, batch_strategy):
        """Test concurrent request handling."""
        num_requests = 20
        inputs = [torch.tensor([float(i)]) for i in range(num_requests)]

        # Submit concurrently
        tasks = [batch_strategy.submit(inp) for inp in inputs]
        results = await asyncio.gather(*tasks)

        # Verify results
        for i, result in enumerate(results):
            expected = inputs[i] * 2
            assert torch.allclose(result, expected)

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, batch_strategy):
        """Test that metrics are correctly tracked."""
        num_requests = 50

        tasks = [batch_strategy.submit(torch.tensor([float(i)])) for i in range(num_requests)]
        await asyncio.gather(*tasks)

        metrics = batch_strategy.get_metrics()

        assert metrics.total_requests == num_requests
        assert metrics.total_batches > 0
        assert metrics.avg_batch_size > 0
        assert metrics.rejected_requests == 0

    @pytest.mark.asyncio
    async def test_queue_full_behavior(self, async_mock_runtime):
        """Test queue full behavior with non-blocking mode."""
        from inferflow.asyncio.batch.dynamic import DynamicBatchStrategy
        from inferflow.asyncio.batch.dynamic import QueueFullError

        # Small queue
        strategy = DynamicBatchStrategy(
            max_batch_size=2,
            max_wait_ms=1000,  # Long wait to fill queue
            queue_size=5,
            block_on_full=False,  # Don't block
        )

        await strategy.start(async_mock_runtime)

        # Fill queue
        tasks = []
        for i in range(10):
            tasks.append(strategy.submit(torch.tensor([float(i)])))

        # Some should be rejected
        results = await asyncio.gather(*tasks, return_exceptions=True)

        rejected_count = sum(1 for r in results if isinstance(r, QueueFullError))

        await strategy.stop()

        assert rejected_count > 0

    @pytest.mark.asyncio
    async def test_batch_size_adaptation(self, large_batch_strategy):
        """Test that batch size adapts to load."""
        # Send bursts with delays
        for _ in range(3):
            tasks = [large_batch_strategy.submit(torch.tensor([float(i)])) for i in range(30)]
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.1)

        metrics = large_batch_strategy.get_metrics()

        # Should use larger batches under load
        assert metrics.avg_batch_size > 5


__all__ = ["TestDynamicBatchStrategy"]
