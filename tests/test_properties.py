# SPDX-License-Identifier: Apache-2.0
"""Property tests on random float64 matrices (tolerance 1e-10).

Step sizes are ``beta = u / (z . z)`` with ``u`` in (0.05, 1), the normalisation that keeps
each update a contraction, so values stay moderate and rounding stays far below the tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from stark.reference import (
    Entry,
    FloatArray,
    delete_entry_by_replay,
    delete_entry_one_step,
    group_by_source,
    remove_source,
    replay,
    trace_query,
)

TOL = 1e-10
SEEDS = range(40)


def _random_case(seed: int) -> tuple[FloatArray, list[Entry]]:
    rng = np.random.default_rng(seed)
    d_out = int(rng.integers(1, 5))
    d_in = int(rng.integers(2, 9))
    n = int(rng.integers(2, 13))
    w0 = rng.normal(size=(d_out, d_in))
    entries = []
    for k in range(n):
        z = rng.normal(size=d_in)
        y = rng.normal(size=d_out)
        beta = float(rng.uniform(0.05, 1.0) / (z @ z))
        entries.append(Entry(source=f"s{k % 3}", z=z, y=y, beta=beta))
    return w0, entries


@pytest.mark.parametrize("seed", SEEDS)
def test_one_step_removal_equals_replay_without_the_entry(seed: int) -> None:
    w0, entries = _random_case(seed)
    live = replay(w0, entries)
    for i in range(len(entries)):
        one_step = delete_entry_one_step(live.weights, entries, live.gaps, i)
        reference = delete_entry_by_replay(w0, entries, i)
        np.testing.assert_allclose(one_step, reference, rtol=0, atol=TOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_trace_shares_add_up_to_the_answer(seed: int) -> None:
    w0, entries = _random_case(seed)
    q: FloatArray = np.random.default_rng(seed + 1000).normal(size=(w0.shape[1],))
    live = replay(w0, entries)

    trace = trace_query(entries, q)

    np.testing.assert_allclose(trace.reconstruct(w0), live.weights @ q, rtol=0, atol=TOL)
    grouped = group_by_source(entries, trace.shares)
    np.testing.assert_allclose(
        w0 @ trace.r0 + sum(grouped.values()), live.weights @ q, rtol=0, atol=TOL
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_source_removal_with_an_unchanged_teacher_equals_replay_of_the_rest(seed: int) -> None:
    w0, entries = _random_case(seed)
    removal = remove_source(w0, entries, "s0", lambda entry: entry.y)

    rest = [entry for entry in entries if entry.source != "s0"]
    np.testing.assert_allclose(removal.weights, replay(w0, rest).weights, rtol=0, atol=TOL)
    assert removal.recomputed == []
