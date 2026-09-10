"""/analysis: outer-joins enrichment_gpu_activity for both sides, and the
inference-ts vs. verification-ts window filters (run cohorts).

Ported from ledger/tests/test_gpu_analysis_join.py and
ledger/tests/test_analysis_inference_window.py: same assertions, seeded via
conftest's seed_event_pair helper instead of ledger CRUD posts, and the GPU
row inserted into enrichment_gpu_activity instead of gpu_activity.
"""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import EnrichmentGpuActivity
from conftest import seed_event_pair


async def test_analysis_includes_gpu_sides(client, engine):
    # enrichment_gpu_activity has no FK tying it to the tables client's
    # TRUNCATE covers, so a leftover row from another module could otherwise
    # be picked up by this test's query — wipe it explicitly before seeding.
    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "TRUNCATE TABLE enrichment_gpu_activity RESTART IDENTITY CASCADE"
        )

    TestSession = async_sessionmaker(engine, expire_on_commit=False)
    async with TestSession() as session:
        inf_id, ver_id = await seed_event_pair(
            session,
            model_name="m-an",
            inf_ts=datetime.datetime(2026, 7, 30, 0, 0, 10, tzinfo=datetime.timezone.utc),
            ver_ts=datetime.datetime(2026, 7, 30, 0, 1, 0, tzinfo=datetime.timezone.utc),
        )

        # Insert enrichment_gpu_activity for the prover side only.
        session.add(
            EnrichmentGpuActivity(
                event_type="inference",
                event_id=inf_id,
                window_s=10.0,
                sample_count=50,
                tensor_active_time_s=4.2,
                sm_occupancy_mean=0.33,
                pipe_activity_s={"DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE": 3.9},
                concurrent_events=0,
                enriched_at=datetime.datetime.now(datetime.timezone.utc),
            )
        )
        await session.commit()

    r = await client.get("/analysis")
    assert r.status_code == 200, r.text
    row = next(item for item in r.json()["items"] if item["id"] == str(ver_id))
    assert row["prover_gpu"]["tensor_active_time_s"] == 4.2
    assert row["prover_gpu"]["sample_count"] == 50
    assert row["verify_gpu"] is None


async def test_inference_window_matches_run_cohort(client, engine):
    TestSession = async_sessionmaker(engine, expire_on_commit=False)
    async with TestSession() as session:
        # In the run window (10:00:00-10:00:30), verified 10 minutes later.
        _, in_window = await seed_event_pair(
            session,
            model_name="m-win",
            inf_ts=datetime.datetime(2026, 8, 3, 10, 0, 10, tzinfo=datetime.timezone.utc),
            ver_ts=datetime.datetime(2026, 8, 3, 10, 10, 0, tzinfo=datetime.timezone.utc),
        )
        # Before the window, but verified DURING it — must not leak into the
        # cohort.
        _, before_window = await seed_event_pair(
            session,
            model_name="m-win",
            inf_ts=datetime.datetime(2026, 8, 3, 9, 59, 0, tzinfo=datetime.timezone.utc),
            ver_ts=datetime.datetime(2026, 8, 3, 10, 0, 15, tzinfo=datetime.timezone.utc),
        )

    r = await client.get(
        "/analysis",
        params={
            "inference_from_ts": "2026-08-03T10:00:00Z",
            "inference_to_ts": "2026-08-03T10:00:30Z",
        },
    )
    assert r.status_code == 200, r.text
    ids = {item["id"] for item in r.json()["items"]}
    assert str(in_window) in ids
    assert str(before_window) not in ids

    # The verification-ts filter over the same window sees the opposite set:
    # the drained-later event is invisible, the verified-during one appears.
    r = await client.get(
        "/analysis",
        params={
            "from_ts": "2026-08-03T10:00:00Z",
            "to_ts": "2026-08-03T10:00:30Z",
        },
    )
    assert r.status_code == 200, r.text
    ids = {item["id"] for item in r.json()["items"]}
    assert str(in_window) not in ids
    assert str(before_window) in ids
