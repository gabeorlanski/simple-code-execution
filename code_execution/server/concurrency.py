"""Concurrency management for the FastAPI server."""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class QueuedExecution:
    """Represents a queued execution request."""
    
    execution_id: str
    priority: int
    queued_time: float
    future: asyncio.Future
    

class ConcurrencyLimiter:
    """Manages execution concurrency with priority queuing."""
    
    def __init__(self, max_concurrency: int = 10):
        """
        Initialize the concurrency limiter.
        
        Args:
            max_concurrency: Maximum number of concurrent executions
        """
        self.max_concurrency = max_concurrency
        self._current_executions = 0
        self._execution_queue: List[QueuedExecution] = []
        self._lock = asyncio.Lock()
        self._execution_counter = 0
        
    async def acquire(self, priority: int = 5) -> str:
        """
        Acquire a slot for execution.
        
        Args:
            priority: Execution priority (1=highest, 10=lowest)
            
        Returns:
            Execution ID for tracking
        """
        async with self._lock:
            self._execution_counter += 1
            execution_id = f"exec_{self._execution_counter}"
            
            if self._current_executions < self.max_concurrency:
                # Slot available immediately
                self._current_executions += 1
                logger.debug(
                    "Acquired execution slot immediately",
                    execution_id=execution_id,
                    current_executions=self._current_executions,
                    max_concurrency=self.max_concurrency
                )
                return execution_id
            else:
                # Need to queue the request
                future = asyncio.Future()
                queued_execution = QueuedExecution(
                    execution_id=execution_id,
                    priority=priority,
                    queued_time=time.time(),
                    future=future
                )
                
                # Insert in priority order (lower number = higher priority)
                insert_idx = 0
                for i, queued in enumerate(self._execution_queue):
                    if priority <= queued.priority:
                        insert_idx = i
                        break
                    insert_idx = i + 1
                    
                self._execution_queue.insert(insert_idx, queued_execution)
                
                logger.debug(
                    "Queued execution request",
                    execution_id=execution_id,
                    priority=priority,
                    queue_position=insert_idx + 1,
                    queue_size=len(self._execution_queue)
                )
                
        # Wait for our turn (outside the lock)
        await future
        return execution_id
        
    async def release(self, execution_id: str) -> None:
        """
        Release an execution slot.
        
        Args:
            execution_id: The execution ID to release
        """
        async with self._lock:
            self._current_executions -= 1
            logger.debug(
                "Released execution slot",
                execution_id=execution_id,
                current_executions=self._current_executions,
                queue_size=len(self._execution_queue)
            )
            
            # Process next item in queue if any
            if self._execution_queue:
                next_execution = self._execution_queue.pop(0)
                self._current_executions += 1
                
                logger.debug(
                    "Promoting queued execution",
                    next_execution_id=next_execution.execution_id,
                    wait_time=time.time() - next_execution.queued_time,
                    remaining_queue_size=len(self._execution_queue)
                )
                
                # Signal the waiting coroutine
                next_execution.future.set_result(None)
                
    def get_queue_info(self) -> Dict[str, int]:
        """
        Get current queue information.
        
        Returns:
            Dictionary with queue statistics
        """
        return {
            "current_executions": self._current_executions,
            "max_concurrency": self.max_concurrency,
            "queue_size": len(self._execution_queue),
            "available_slots": max(0, self.max_concurrency - self._current_executions)
        }
        
    def get_queue_position(self, execution_id: str) -> Optional[int]:
        """
        Get the position of an execution in the queue.
        
        Args:
            execution_id: The execution ID to check
            
        Returns:
            Queue position (1-based) or None if not in queue
        """
        for i, queued in enumerate(self._execution_queue):
            if queued.execution_id == execution_id:
                return i + 1
        return None
        
    def estimate_wait_time(self, queue_position: int) -> float:
        """
        Estimate wait time based on queue position.
        
        Args:
            queue_position: Position in queue (1-based)
            
        Returns:
            Estimated wait time in seconds
        """
        # Simple estimation: assume average execution time of 10 seconds
        # and that slots become available sequentially
        avg_execution_time = 10.0
        slots_ahead = max(0, queue_position - self.max_concurrency)
        return slots_ahead * (avg_execution_time / self.max_concurrency)
        
    async def set_max_concurrency(self, new_max: int) -> None:
        """
        Update the maximum concurrency limit.
        
        Args:
            new_max: New maximum concurrency limit
        """
        async with self._lock:
            old_max = self.max_concurrency
            self.max_concurrency = new_max
            
            logger.info(
                "Updated max concurrency",
                old_max=old_max,
                new_max=new_max,
                current_executions=self._current_executions
            )
            
            # If we increased capacity, process queued items
            if new_max > old_max:
                additional_slots = new_max - self._current_executions
                while additional_slots > 0 and self._execution_queue:
                    next_execution = self._execution_queue.pop(0)
                    self._current_executions += 1
                    additional_slots -= 1
                    
                    logger.debug(
                        "Processing queued execution due to increased capacity",
                        execution_id=next_execution.execution_id
                    )
                    
                    next_execution.future.set_result(None)


# Global concurrency limiter instance
_global_limiter: Optional[ConcurrencyLimiter] = None


def get_concurrency_limiter() -> ConcurrencyLimiter:
    """
    Get the global concurrency limiter instance.
    
    Returns:
        The global ConcurrencyLimiter instance
    """
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = ConcurrencyLimiter()
    return _global_limiter


async def set_max_concurrency(max_concurrency: int) -> None:
    """
    Set the global maximum concurrency.
    
    Args:
        max_concurrency: New maximum concurrency limit
    """
    limiter = get_concurrency_limiter()
    await limiter.set_max_concurrency(max_concurrency)