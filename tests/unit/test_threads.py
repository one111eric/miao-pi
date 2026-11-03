import asyncio

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import AgentGraph, build_agent_graph
from app.agent.threads import ThreadBusyError, ThreadStore
from tests.fakes import EchoChatModel


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_store(clock: FakeClock, *, max_threads: int = 100) -> ThreadStore:
    return ThreadStore(ttl=3600, max_threads=max_threads, clock=clock)


async def invoke(graph: AgentGraph, thread_id: str) -> None:
    await graph.ainvoke(
        {"messages": [HumanMessage(content="Hello")]},
        config={"configurable": {"thread_id": thread_id}},
    )


async def run(store: ThreadStore, graph: AgentGraph, thread_id: str) -> None:
    async with store.use(thread_id):
        await invoke(graph, thread_id)


async def has_checkpoint(store: ThreadStore, thread_id: str) -> bool:
    saved = await store.checkpointer.aget_tuple(
        {"configurable": {"thread_id": thread_id}}
    )
    return saved is not None


@pytest.mark.anyio
async def test_threads_unused_past_the_ttl_are_deleted(clock: FakeClock) -> None:
    store = make_store(clock)
    graph = build_agent_graph(EchoChatModel(), checkpointer=store.checkpointer)

    await run(store, graph, "a")
    clock.now = 100
    await run(store, graph, "b")
    clock.now = 3700  # "a" unused for 3700s, "b" for exactly the ttl
    await run(store, graph, "c")

    assert not await has_checkpoint(store, "a")
    assert await has_checkpoint(store, "b")
    assert await has_checkpoint(store, "c")


@pytest.mark.anyio
async def test_least_recently_used_threads_go_first_beyond_the_cap(
    clock: FakeClock,
) -> None:
    store = make_store(clock, max_threads=2)
    graph = build_agent_graph(EchoChatModel(), checkpointer=store.checkpointer)

    for thread_id in ["a", "b", "a", "c"]:
        clock.now += 1
        await run(store, graph, thread_id)

    assert await has_checkpoint(store, "a")
    assert not await has_checkpoint(store, "b")
    assert await has_checkpoint(store, "c")


@pytest.mark.anyio
async def test_an_expired_thread_starts_fresh(clock: FakeClock) -> None:
    model = EchoChatModel()
    store = make_store(clock)
    graph = build_agent_graph(model, checkpointer=store.checkpointer)

    await run(store, graph, "a")
    clock.now = 4000
    async with store.use("a"):
        assert not await has_checkpoint(store, "a")
        await invoke(graph, "a")

    assert [message.type for message in model.calls[1]] == ["system", "human"]


@pytest.mark.anyio
async def test_a_second_run_on_a_busy_thread_is_rejected(clock: FakeClock) -> None:
    store = make_store(clock)

    async with store.use("a"):
        with pytest.raises(ThreadBusyError):
            async with store.use("a"):
                pass
        async with store.use("b"):  # other threads are unaffected
            pass

    async with store.use("a"):  # free again once the first run ends
        pass


@pytest.mark.anyio
async def test_a_failed_run_releases_the_thread(clock: FakeClock) -> None:
    store = make_store(clock)

    with pytest.raises(RuntimeError):
        async with store.use("a"):
            raise RuntimeError("run failed")

    async with store.use("a"):
        pass


@pytest.mark.anyio
async def test_a_thread_in_use_is_not_evicted_past_its_ttl(clock: FakeClock) -> None:
    store = make_store(clock)
    graph = build_agent_graph(EchoChatModel(), checkpointer=store.checkpointer)
    await run(store, graph, "a")

    async with store.use("a"):
        clock.now = 4000  # past the ttl while "a" is still running
        await run(store, graph, "b")
        assert await has_checkpoint(store, "a")

    # Finishing counts as use, so "a" gets a fresh ttl from 4000.
    clock.now = 7000
    await run(store, graph, "c")
    assert await has_checkpoint(store, "a")


@pytest.mark.anyio
async def test_the_cap_is_restored_as_soon_as_runs_finish(clock: FakeClock) -> None:
    store = make_store(clock, max_threads=1)
    graph = build_agent_graph(EchoChatModel(), checkpointer=store.checkpointer)

    async with store.use("a"):
        await invoke(graph, "a")
        clock.now += 1
        async with store.use("b"):  # two threads, both running, cap of one
            await invoke(graph, "b")
            assert await has_checkpoint(store, "a")
            assert await has_checkpoint(store, "b")
        clock.now += 1
        # "b" finished while "a" still runs, so it was the only thread the
        # cap could remove.
        assert not await has_checkpoint(store, "b")

    # Back within the cap without another run starting.
    assert await has_checkpoint(store, "a")
    assert not await has_checkpoint(store, "b")


class YieldingSaver(InMemorySaver):
    """Pauses on every delete, like a checkpointer doing real I/O."""

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.sleep(0)
        await super().adelete_thread(thread_id)


@pytest.mark.anyio
async def test_a_run_finishing_during_eviction_is_not_deleted(
    clock: FakeClock,
) -> None:
    store = make_store(clock)
    store.checkpointer = YieldingSaver()
    graph = build_agent_graph(EchoChatModel(), checkpointer=store.checkpointer)
    await run(store, graph, "x")

    entered = asyncio.Event()
    release = asyncio.Event()

    async def run_y() -> None:
        async with store.use("y"):
            await invoke(graph, "y")
            entered.set()
            await release.wait()

    clock.now = 1
    task = asyncio.create_task(run_y())
    await asyncio.wait_for(entered.wait(), timeout=2)

    # Both threads are now past the ttl. "y" finishes while the eviction for
    # "z" pauses on deleting "x"; that eviction must not act on a stale view
    # of "y", which is fresh again once it finishes.
    clock.now = 5000
    release.set()
    async with store.use("z"):
        pass
    await asyncio.wait_for(task, timeout=2)

    assert not await has_checkpoint(store, "x")
    assert await has_checkpoint(store, "y")


@pytest.mark.anyio
async def test_a_cancelled_run_releases_the_thread_while_cleanup_waits(
    clock: FakeClock,
) -> None:
    store = make_store(clock)
    entered = asyncio.Event()

    async def hold_thread() -> None:
        async with store.use("a"):
            entered.set()
            await asyncio.Event().wait()  # runs until cancelled

    task = asyncio.create_task(hold_thread())
    await asyncio.wait_for(entered.wait(), timeout=2)

    # Hold the store's lock so the cleanup has to wait for it, then cancel the
    # run twice: once to end it, and again while its cleanup is waiting.
    async with store._lock:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Free the moment the task finished, without waiting for the lock.
        assert "a" not in store._active

    async with store.use("a"):
        pass


class FailingOnceSaver(InMemorySaver):
    """Fails the first delete, like a checkpointer with a transient error."""

    failed: bool = False

    async def adelete_thread(self, thread_id: str) -> None:
        if not self.failed:
            self.failed = True
            raise RuntimeError("storage unavailable")
        await super().adelete_thread(thread_id)


@pytest.mark.anyio
async def test_a_failed_delete_keeps_the_thread_tracked_for_a_retry(
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = make_store(clock, max_threads=1)
    store.checkpointer = FailingOnceSaver()
    graph = build_agent_graph(EchoChatModel(), checkpointer=store.checkpointer)

    await run(store, graph, "a")
    clock.now += 1
    await run(store, graph, "b")  # evicting "a" fails, but this run succeeds

    assert await has_checkpoint(store, "a")
    assert "Failed to evict conversation thread" in caplog.text

    clock.now += 1
    await run(store, graph, "c")  # "a" is still tracked, so it's retried

    assert not await has_checkpoint(store, "a")
    assert not await has_checkpoint(store, "b")
    assert await has_checkpoint(store, "c")


@pytest.mark.parametrize(
    ("ttl", "max_threads"),
    [(0, 10), (60, 0)],
)
def test_rejects_invalid_limits(ttl: float, max_threads: int) -> None:
    with pytest.raises(ValueError):
        ThreadStore(ttl=ttl, max_threads=max_threads)
