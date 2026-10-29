# SPDX-License-Identifier: Apache-2.0
"""Hand-checked toy example in float64 numpy (tolerance 1e-9).

A 2x3 matrix W starts at zero, beta = 0.5, and each entry applies
``W <- W + beta * (y - W z) z^T``. Entries (z = address, y = teacher answer):

    E1: z=(1,0,0)      y=(+1,0)
    E2: z=(0,1,0)      y=(0,+1)
    E3: z=(1,0,0)      y=(-1,0)
    E4: z=(0.6,0,0.8)  y=(-0.6,0) while E3's source is visible to the teacher,
                       y=(+0.6,0) when it is hidden

The expected numbers below were worked out by hand. Do NOT edit them to make a test pass; a
failure means the maths broke.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt

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

ATOL = 1e-9
BETA = 0.5
Q1 = np.array([1.0, 0.0, 0.0])
Q2 = np.array([0.6, 0.0, 0.8])
W0 = np.zeros((2, 3))


def _v(*values: float) -> FloatArray:
    return np.array(values, dtype=np.float64)


def _close(actual: npt.ArrayLike, expected: npt.ArrayLike) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0,
        atol=ATOL,
    )


def _ledger(y4_first: float) -> list[Entry]:
    return [
        Entry("s1", _v(1, 0, 0), _v(1, 0), BETA),
        Entry("s2", _v(0, 1, 0), _v(0, 1), BETA),
        Entry("s3", _v(1, 0, 0), _v(-1, 0), BETA),
        Entry("s4", _v(0.6, 0, 0.8), _v(y4_first, 0), BETA),
    ]


def test_final_weights_and_queries() -> None:
    result = replay(W0, _ledger(-0.6))
    _close(result.weights[0], [-0.385, 0.0, -0.18])
    _close(result.weights[1], [0.0, 0.5, 0.0])
    _close(result.weights[0] @ Q1, -0.385)
    _close(result.weights[0] @ Q2, -0.375)


def test_trace_of_q1() -> None:
    ledger = _ledger(-0.6)
    live = replay(W0, ledger)
    trace = trace_query(ledger, Q1)

    first_component = [share[0] for share in trace.shares]  # E1, E2, E3, E4
    _close(first_component, [0.205, 0.0, -0.41, -0.18])
    _close(sum(first_component), -0.385)
    _close(trace.reconstruct(W0), live.weights @ Q1)


def test_trace_grouped_by_source() -> None:
    ledger = [
        dataclasses.replace(e, source=s)
        for e, s in zip(_ledger(-0.6), ["A", "A", "B", "C"], strict=True)
    ]
    grouped = group_by_source(ledger, trace_query(ledger, Q1).shares)

    assert list(grouped) == ["A", "B", "C"]
    _close(grouped["A"][0], 0.205 + 0.0)
    _close(grouped["B"][0], -0.41)
    _close(grouped["C"][0], -0.18)


def test_delete_only_e3_leaves_a_leak() -> None:
    ledger = _ledger(-0.6)
    live = replay(W0, ledger)

    deleted = delete_entry_one_step(live.weights, ledger, live.gaps, 2)

    _close(deleted[0], [0.23, 0.0, -0.36])
    _close(deleted[1], [0.0, 0.5, 0.0])
    _close(deleted[0] @ Q1, 0.23)
    _close(deleted[0] @ Q2, -0.15)
    _close(deleted, delete_entry_by_replay(W0, ledger, 2))


def test_remove_e3_as_source_matches_a_run_that_never_saw_it() -> None:
    ledger = _ledger(-0.6)

    def teacher_without_s3(entry: Entry) -> FloatArray:
        return _v(0.6, 0) if entry.source == "s4" else entry.y

    removal = remove_source(W0, ledger, "s3", teacher_without_s3)

    assert removal.removed == [2]
    assert removal.recomputed == [3]
    _close(removal.weights[0], [0.59, 0.0, 0.12])
    _close(removal.weights[0] @ Q1, 0.59)
    _close(removal.weights[0] @ Q2, 0.45)

    # A run that never saw E3: E1, E2 and E4 with the source-hidden teacher answer.
    clean_ledger = [_ledger(0.6)[i] for i in (0, 1, 3)]
    _close(removal.weights, replay(W0, clean_ledger).weights)

    # Deleting E3 alone left Q2 at -0.15; the true clean answer is +0.45.
    assert abs(-0.15 - removal.weights[0] @ Q2) > 0.5
