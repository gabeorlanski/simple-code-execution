"""Unit tests for concurrency management."""

import asyncio
from unittest.mock import patch

import pytest

from code_execution.server.concurrency import ConcurrencyLimiter
from code_execution.server.concurrency import get_concurrency_limiter
from code_execution.server.concurrency import set_max_concurrency


class TestConcurrencyLimiter:
    """Test cases for the ConcurrencyLimiter class."""

    def test_init_default_concurrency(self):
        """Test initialization with default concurrency."""
        limiter = ConcurrencyLimiter()

        assert limiter.max_concurrency == 10
        assert limiter._current_executions == 0
        assert len(limiter._execution_queue) == 0

    def test_init_custom_concurrency(self):
        """Test initialization with custom concurrency."""
        limiter = ConcurrencyLimiter(max_concurrency=5)

        assert limiter.max_concurrency == 5
        assert limiter._current_executions == 0

    @pytest.mark.asyncio
    async def test_acquire_slot_available(self):
        """Test acquiring slot when slots are available."""
        limiter = ConcurrencyLimiter(max_concurrency=2)

        execution_id = await limiter.acquire()

        assert execution_id.startswith("exec_")
        assert limiter._current_executions == 1
        assert len(limiter._execution_queue) == 0

    @pytest.mark.asyncio
    async def test_acquire_multiple_slots(self):
        """Test acquiring multiple slots up to limit."""
        limiter = ConcurrencyLimiter(max_concurrency=2)

        exec_id1 = await limiter.acquire()
        exec_id2 = await limiter.acquire()

        assert exec_id1 != exec_id2
        assert limiter._current_executions == 2
        assert len(limiter._execution_queue) == 0

    @pytest.mark.asyncio
    async def test_acquire_queue_when_at_capacity(self):
        """Test that requests are queued when at capacity."""
        limiter = ConcurrencyLimiter(max_concurrency=1)

        # First request should succeed immediately
        exec_id1 = await limiter.acquire()
        assert limiter._current_executions == 1

        # Second request should be queued
        acquire_task = asyncio.create_task(limiter.acquire())

        # Give some time for the task to start
        await asyncio.sleep(0.01)

        # Should be queued, not completed
        assert not acquire_task.done()
        assert len(limiter._execution_queue) == 1
        assert limiter._current_executions == 1

        # Release first slot
        await limiter.release(exec_id1)

        # Second request should now complete
        exec_id2 = await acquire_task
        assert exec_id2.startswith("exec_")
        assert limiter._current_executions == 1
        assert len(limiter._execution_queue) == 0

    @pytest.mark.asyncio
    async def test_fifo_queue_ordering(self):
        """Test that requests are processed in FIFO order."""
        limiter = ConcurrencyLimiter(max_concurrency=1)

        # Fill capacity
        exec_id1 = await limiter.acquire()

        # Queue requests in order
        first_task = asyncio.create_task(limiter.acquire())
        second_task = asyncio.create_task(limiter.acquire())
        third_task = asyncio.create_task(limiter.acquire())

        await asyncio.sleep(0.01)  # Let them queue

        # Verify queue size
        assert len(limiter._execution_queue) == 3

        # Release slot and verify first queued gets it
        await limiter.release(exec_id1)

        first_id = await first_task
        assert first_id.startswith("exec_")

        # Clean up remaining tasks
        await limiter.release(first_id)
        second_id = await second_task
        await limiter.release(second_id)
        third_id = await third_task
        await limiter.release(third_id)

    @pytest.mark.asyncio
    async def test_release_slot(self):
        """Test releasing an execution slot."""
        limiter = ConcurrencyLimiter(max_concurrency=2)

        exec_id = await limiter.acquire()
        assert limiter._current_executions == 1

        await limiter.release(exec_id)
        assert limiter._current_executions == 0

    def test_get_queue_info(self):
        """Test getting queue information."""
        limiter = ConcurrencyLimiter(max_concurrency=5)
        limiter._current_executions = 3

        info = limiter.get_queue_info()

        expected = {
            "current_executions": 3,
            "max_concurrency": 5,
            "queue_size": 0,
            "available_slots": 2,
        }
        assert info == expected

    def test_get_queue_position_not_in_queue(self):
        """Test getting queue position for ID not in queue."""
        limiter = ConcurrencyLimiter()

        position = limiter.get_queue_position("nonexistent_id")

        assert position is None

    @pytest.mark.asyncio
    async def test_set_max_concurrency_increase(self):
        """Test increasing max concurrency processes queued items."""
        limiter = ConcurrencyLimiter(max_concurrency=1)

        # Fill capacity and queue a request
        exec_id1 = await limiter.acquire()
        queued_task = asyncio.create_task(limiter.acquire())

        await asyncio.sleep(0.01)  # Let it queue
        assert len(limiter._execution_queue) == 1

        # Increase capacity
        await limiter.set_max_concurrency(2)

        # Queued task should complete
        exec_id2 = await queued_task
        assert exec_id2.startswith("exec_")
        assert limiter._current_executions == 2
        assert len(limiter._execution_queue) == 0

    @pytest.mark.asyncio
    async def test_set_max_concurrency_decrease(self):
        """Test decreasing max concurrency."""
        limiter = ConcurrencyLimiter(max_concurrency=5)

        # Acquire some slots
        exec_id1 = await limiter.acquire()
        exec_id2 = await limiter.acquire()

        # Decrease capacity
        await limiter.set_max_concurrency(3)

        assert limiter.max_concurrency == 3
        assert limiter._current_executions == 2  # Current executions unchanged


class TestGlobalLimiter:
    """Test cases for global limiter functions."""

    def test_get_concurrency_limiter_singleton(self):
        """Test that get_concurrency_limiter returns singleton."""
        limiter1 = get_concurrency_limiter()
        limiter2 = get_concurrency_limiter()

        assert limiter1 is limiter2

    @pytest.mark.asyncio
    async def test_set_max_concurrency_global(self):
        """Test setting global max concurrency."""
        with patch.object(
            ConcurrencyLimiter, "set_max_concurrency"
        ) as mock_set:
            await set_max_concurrency(15)

            mock_set.assert_called_once_with(15)


class TestConcurrencyIntegration:
    """Integration tests for concurrency features."""

    @pytest.mark.asyncio
    async def test_concurrent_execution_limiting(self):
        """Test that concurrent executions are properly limited."""
        limiter = ConcurrencyLimiter(max_concurrency=2)

        # Track execution starts
        started = []
        completed = []

        async def mock_execution(task_id: int):
            """Mock execution that tracks when it starts and completes."""
            exec_id = await limiter.acquire()
            try:
                started.append(task_id)
                await asyncio.sleep(0.1)
                completed.append(task_id)
                return f"result_{task_id}"
            finally:
                await limiter.release(exec_id)

        # Start 4 tasks concurrently
        tasks = [asyncio.create_task(mock_execution(i)) for i in range(4)]

        # Wait a short time to let first batch start
        await asyncio.sleep(0.05)

        # Should have only 2 started (concurrency limit)
        assert len(started) == 2
        assert len(completed) == 0

        # Wait for first batch to complete
        await asyncio.sleep(0.1)

        # Now should have 2 completed and 2 more started
        assert len(completed) == 2
        await asyncio.sleep(0.05)  # Let second batch start
        assert len(started) == 4

        # Wait for all to complete
        await asyncio.gather(*tasks)
        assert len(completed) == 4

    @pytest.mark.asyncio
    async def test_fifo_execution_order(self):
        """Test that executions run in FIFO order."""
        limiter = ConcurrencyLimiter(max_concurrency=1)

        execution_order = []

        async def fifo_execution(task_id: int):
            """Mock execution that records its task ID."""
            exec_id = await limiter.acquire()
            try:
                execution_order.append(task_id)
                await asyncio.sleep(0.01)
            finally:
                await limiter.release(exec_id)

        # Start tasks in order
        tasks = [
            asyncio.create_task(fifo_execution(1)),
            asyncio.create_task(fifo_execution(2)),
            asyncio.create_task(fifo_execution(3)),
        ]

        # Wait for all to complete
        await asyncio.gather(*tasks)

        # Should execute in FIFO order: 1, 2, 3
        assert execution_order == [1, 2, 3]
