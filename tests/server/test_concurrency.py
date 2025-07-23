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
        
        execution_id = await limiter.acquire(priority=5)
        
        assert execution_id.startswith("exec_")
        assert limiter._current_executions == 1
        assert len(limiter._execution_queue) == 0

    @pytest.mark.asyncio
    async def test_acquire_multiple_slots(self):
        """Test acquiring multiple slots up to limit."""
        limiter = ConcurrencyLimiter(max_concurrency=2)
        
        exec_id1 = await limiter.acquire(priority=5)
        exec_id2 = await limiter.acquire(priority=5)
        
        assert exec_id1 != exec_id2
        assert limiter._current_executions == 2
        assert len(limiter._execution_queue) == 0

    @pytest.mark.asyncio
    async def test_acquire_queue_when_at_capacity(self):
        """Test that requests are queued when at capacity."""
        limiter = ConcurrencyLimiter(max_concurrency=1)
        
        # First request should succeed immediately
        exec_id1 = await limiter.acquire(priority=5)
        assert limiter._current_executions == 1
        
        # Second request should be queued
        acquire_task = asyncio.create_task(limiter.acquire(priority=5))
        
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
    async def test_priority_queue_ordering(self):
        """Test that higher priority requests are processed first."""
        limiter = ConcurrencyLimiter(max_concurrency=1)
        
        # Fill capacity
        exec_id1 = await limiter.acquire(priority=5)
        
        # Queue requests with different priorities
        low_priority_task = asyncio.create_task(limiter.acquire(priority=8))
        high_priority_task = asyncio.create_task(limiter.acquire(priority=2))
        medium_priority_task = asyncio.create_task(limiter.acquire(priority=5))
        
        await asyncio.sleep(0.01)  # Let them queue
        
        # Verify queue order (should be high, medium, low priority)
        assert len(limiter._execution_queue) == 3
        assert limiter._execution_queue[0].priority == 2  # highest priority
        assert limiter._execution_queue[1].priority == 5  # medium priority
        assert limiter._execution_queue[2].priority == 8  # lowest priority
        
        # Release slot and verify high priority gets it
        await limiter.release(exec_id1)
        
        high_priority_id = await high_priority_task
        assert high_priority_id.startswith("exec_")
        
        # Clean up remaining tasks
        await limiter.release(high_priority_id)
        await medium_priority_task
        await low_priority_task

    @pytest.mark.asyncio
    async def test_release_slot(self):
        """Test releasing an execution slot."""
        limiter = ConcurrencyLimiter(max_concurrency=2)
        
        exec_id = await limiter.acquire(priority=5)
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
            "available_slots": 2
        }
        assert info == expected

    def test_get_queue_position_not_in_queue(self):
        """Test getting queue position for ID not in queue."""
        limiter = ConcurrencyLimiter()
        
        position = limiter.get_queue_position("nonexistent_id")
        
        assert position is None

    def test_estimate_wait_time(self):
        """Test wait time estimation."""
        limiter = ConcurrencyLimiter(max_concurrency=2)
        
        # Position 1 should have minimal wait time
        wait_time = limiter.estimate_wait_time(1)
        assert wait_time == 0.0
        
        # Position 5 should have some wait time
        wait_time = limiter.estimate_wait_time(5)
        assert wait_time > 0

    @pytest.mark.asyncio
    async def test_set_max_concurrency_increase(self):
        """Test increasing max concurrency processes queued items."""
        limiter = ConcurrencyLimiter(max_concurrency=1)
        
        # Fill capacity and queue a request
        exec_id1 = await limiter.acquire(priority=5)
        queued_task = asyncio.create_task(limiter.acquire(priority=5))
        
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
        exec_id1 = await limiter.acquire(priority=5)
        exec_id2 = await limiter.acquire(priority=5)
        
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
        with patch.object(ConcurrencyLimiter, 'set_max_concurrency') as mock_set:
            await set_max_concurrency(15)
            
            mock_set.assert_called_once_with(15)


class TestConcurrencyIntegration:
    """Integration tests for concurrency features."""

    @pytest.mark.asyncio
    async def test_concurrent_execution_limiting(self):
        """Test that concurrent executions are properly limited."""
        limiter = ConcurrencyLimiter(max_concurrency=2)
        
        # Track completion order
        completed = []
        
        async def mock_execution(exec_id: str, delay: float):
            """Mock execution that takes some time."""
            await asyncio.sleep(delay)
            completed.append(exec_id)
            return f"result_{exec_id}"
        
        # Start 4 tasks but only 2 should run concurrently
        tasks = []
        for i in range(4):
            exec_id = await limiter.acquire(priority=5)
            task = asyncio.create_task(mock_execution(exec_id, 0.1))
            tasks.append((exec_id, task))
        
        # Wait for first batch to complete
        await asyncio.sleep(0.15)
        
        # Should have 2 completed
        assert len(completed) == 2
        
        # Release first two slots
        for i in range(2):
            exec_id, _ = tasks[i]
            await limiter.release(exec_id)
        
        # Wait for second batch
        await asyncio.sleep(0.15)
        
        # All should be completed now
        assert len(completed) == 4
        
        # Release remaining slots
        for i in range(2, 4):
            exec_id, _ = tasks[i]
            await limiter.release(exec_id)

    @pytest.mark.asyncio
    async def test_priority_execution_order(self):
        """Test that higher priority executions run first."""
        limiter = ConcurrencyLimiter(max_concurrency=1)
        
        execution_order = []
        
        async def priority_execution(priority: int):
            """Mock execution that records its priority."""
            exec_id = await limiter.acquire(priority=priority)
            try:
                execution_order.append(priority)
                await asyncio.sleep(0.01)
            finally:
                await limiter.release(exec_id)
        
        # Start tasks in reverse priority order
        tasks = [
            asyncio.create_task(priority_execution(8)),  # low priority
            asyncio.create_task(priority_execution(3)),  # high priority
            asyncio.create_task(priority_execution(6)),  # medium priority
        ]
        
        # Wait for all to complete
        await asyncio.gather(*tasks)
        
        # Should execute in priority order: 3, 6, 8
        assert execution_order == [8, 3, 6]  # First one runs immediately, then by priority