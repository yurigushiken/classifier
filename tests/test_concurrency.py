from __future__ import annotations

import asyncio

from classifier_pipeline.phase3_pilot import run_with_semaphore


def test_run_with_semaphore_limits_concurrency():
    active = 0
    max_active = 0

    async def worker(item: int) -> int:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return item

    items = list(range(10))
    results = run_with_semaphore(items, worker, max_concurrent=3)

    assert sorted(results) == items
    assert max_active <= 3
