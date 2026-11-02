import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)


class ThreadBusyError(Exception):
    """Raised when a thread already has a run in progress."""

    def __init__(self, thread_id: str) -> None:
        super().__init__(f"Thread {thread_id} already has a run in progress")
        self.thread_id = thread_id


class ThreadStore:
    """In-memory conversation checkpoints with a retention cap.

    Each thread runs one request at a time. Idle threads unused for longer
    than ``ttl`` seconds are deleted when a run starts or finishes. When a
    run finishes, the least recently used idle threads beyond
    ``max_threads`` are deleted. Threads with a run in progress are never
    deleted, so the cap can be exceeded by the number of concurrent runs.
    """

    def __init__(
        self,
        ttl: float,
        max_threads: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        if max_threads < 1:
            raise ValueError("max_threads must be at least 1")

        self.checkpointer = InMemorySaver()
        self._ttl = ttl
        self._max_threads = max_threads
        self._clock = clock
        # Ordered from least to most recently used.
        self._last_used: OrderedDict[str, float] = OrderedDict()
        self._active: set[str] = set()
        # Serializes starting runs and eviction. A finishing run releases its
        # thread without it (so release can't be cancelled or delayed), which
        # is why eviction re-reads the state after every await.
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def use(self, thread_id: str) -> AsyncIterator[None]:
        """Hold a thread for the duration of one run.

        Raises ``ThreadBusyError`` if the thread already has a run in
        progress. Expired threads are deleted on entry, so an expired thread
        starts fresh instead of loading its old checkpoint.
        """
        async with self._lock:
            if thread_id in self._active:
                raise ThreadBusyError(thread_id)
            now = self._clock()
            await self._evict_expired(now)
            # Nothing awaits after this, so a cancellation can't leave the
            # thread marked active without the release below.
            self._active.add(thread_id)
            self._mark_used(thread_id, now)
        try:
            yield
        finally:
            # Released without awaiting, so the thread is free the moment the
            # run ends, even when the run was cancelled.
            self._active.discard(thread_id)
            self._mark_used(thread_id, self._clock())
            # Best effort: if this is cancelled, a later run cleans up instead.
            async with self._lock:
                await self._evict_expired(self._clock())
                await self._evict_over_cap()

    def _mark_used(self, thread_id: str, now: float) -> None:
        self._last_used[thread_id] = now
        self._last_used.move_to_end(thread_id)

    def _oldest_idle(self) -> str | None:
        return next((t for t in self._last_used if t not in self._active), None)

    async def _evict_expired(self, now: float) -> None:
        while (thread_id := self._oldest_idle()) is not None:
            if now - self._last_used[thread_id] <= self._ttl:
                return  # Every other idle thread was used more recently.
            if not await self._delete(thread_id, reason="ttl"):
                return

    async def _evict_over_cap(self) -> None:
        while len(self._last_used) > self._max_threads:
            thread_id = self._oldest_idle()
            if thread_id is None:
                return  # Every thread has a run in progress.
            if not await self._delete(thread_id, reason="max_threads"):
                return

    async def _delete(self, thread_id: str, *, reason: str) -> bool:
        """Delete a thread's checkpoints, then stop tracking it.

        A failed delete is logged and returns False. The thread stays
        tracked, so a later cleanup retries it, and the run that triggered
        the cleanup isn't failed by it.
        """
        log_fields = {"evicted_thread_id": thread_id, "reason": reason}
        try:
            await self.checkpointer.adelete_thread(thread_id)
        except Exception:
            logger.exception("Failed to evict conversation thread", extra=log_fields)
            return False
        del self._last_used[thread_id]
        logger.info("Evicted conversation thread", extra=log_fields)
        return True
