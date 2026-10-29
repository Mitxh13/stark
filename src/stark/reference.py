# SPDX-License-Identifier: Apache-2.0
"""Float64 numpy reference for the STARK update rule, replay, trace and removal.

This is the oracle for the toy example and the property tests, and later for the real code.
The real learning layer runs in float32 with deterministic kernels; nothing here is used on
that path.

One ledger entry holds an address ``z``, a recorded teacher answer ``y`` and a step size
``beta``. Applying it changes the learning matrix ``W`` by::

    E  = y - W z                  (the gap)
    W <- W + beta * E z^T

Because ``y`` is a recorded constant, the update is affine in ``W``::

    W <- W A + beta * y z^T,      A = I - beta z z^T

which is what makes the exact trace and the exact removal below possible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class Entry:
    """One ledger entry.

    Attributes:
        source: Identifier of the source (for example a document) the entry came from.
        z: Address, shape ``(d_in,)``. Never depends on ``W``.
        y: Recorded teacher answer, shape ``(d_out,)``.
        beta: Step size.
    """

    source: str
    z: FloatArray
    y: FloatArray
    beta: float

    def __post_init__(self) -> None:
        if self.z.ndim != 1 or self.y.ndim != 1:
            raise ValueError("z and y must be 1-D arrays")

    def with_y(self, y: FloatArray) -> Entry:
        """Return a copy of this entry with a different recorded teacher answer."""
        return replace(self, y=y)


@dataclass(frozen=True, eq=False)
class Replay:
    """Result of replaying entries.

    Attributes:
        weights: Final ``W``, shape ``(d_out, d_in)``.
        gaps: ``E_i = y_i - W_{i-1} z_i`` for each entry, in order.
    """

    weights: FloatArray
    gaps: list[FloatArray]


def apply_update(w: FloatArray, entry: Entry) -> tuple[FloatArray, FloatArray]:
    """Apply one entry to ``w``.

    Returns:
        ``(W_new, gap)`` where ``gap = y - W z`` is measured before the update.
    """
    gap: FloatArray = entry.y - w @ entry.z
    return w + entry.beta * np.outer(gap, entry.z), gap


def replay(w0: FloatArray, entries: Sequence[Entry]) -> Replay:
    """Replay ``entries`` in order starting from ``w0``.

    ``w0`` may be the initial matrix or a saved checkpoint, in which case ``entries`` are the
    entries after that checkpoint. The input is never modified.
    """
    w = np.array(w0, dtype=np.float64, copy=True)
    gaps: list[FloatArray] = []
    for entry in entries:
        w, gap = apply_update(w, entry)
        gaps.append(gap)
    return Replay(weights=w, gaps=gaps)


@dataclass(frozen=True, eq=False)
class QueryTrace:
    """Per-entry shares of one answer.

    Attributes:
        shares: ``share_i`` for each entry, in ledger order, each of shape ``(d_out,)``.
        r0: The query pulled back through every entry; the initial matrix contributes
            ``W_0 @ r0``.
    """

    shares: list[FloatArray]
    r0: FloatArray

    def reconstruct(self, w0: FloatArray) -> FloatArray:
        """Return ``W_0 r_0 + sum(shares)``, which equals the traced answer ``W_n q``."""
        total: FloatArray = w0 @ self.r0
        for share in self.shares:
            total = total + share
        return total


def trace_query(entries: Sequence[Entry], q: FloatArray) -> QueryTrace:
    """Trace the answer ``W_n q`` back to each entry.

    Walking the entries backwards::

        r_n     = q
        r_{i-1} = r_i - beta_i z_i (z_i . r_i)        (that is, A_i r_i)
        share_i = beta_i y_i (z_i . r_i)

    gives ``W_n q = W_0 r_0 + sum_i share_i`` exactly.
    """
    r = np.array(q, dtype=np.float64, copy=True)
    shares: list[FloatArray] = []
    for entry in reversed(entries):
        overlap = float(entry.z @ r)
        shares.append(entry.beta * overlap * entry.y)
        r = r - entry.beta * overlap * entry.z
    shares.reverse()
    return QueryTrace(shares=shares, r0=r)


def group_by_source(
    entries: Sequence[Entry], shares: Sequence[FloatArray]
) -> dict[str, FloatArray]:
    """Sum entry shares per source, keeping the order in which sources first appear."""
    if len(entries) != len(shares):
        raise ValueError("entries and shares must have the same length")
    grouped: dict[str, FloatArray] = {}
    for entry, share in zip(entries, shares, strict=True):
        if entry.source in grouped:
            grouped[entry.source] = grouped[entry.source] + share
        else:
            grouped[entry.source] = np.array(share, dtype=np.float64, copy=True)
    return grouped


def propagate_address(entries: Sequence[Entry], i: int) -> FloatArray:
    """Return ``U_i = A_n ... A_{i+1} z_i`` by applying each ``A_j`` as a rank-one update."""
    u = np.array(entries[i].z, dtype=np.float64, copy=True)
    for later in entries[i + 1 :]:
        u = u - later.beta * float(later.z @ u) * later.z
    return u


def delete_entry_one_step(
    w_final: FloatArray,
    entries: Sequence[Entry],
    gaps: Sequence[FloatArray],
    i: int,
) -> FloatArray:
    """Remove entry ``i`` from the final matrix in one step, without replaying::

        W' = W - beta_i E_i U_i^T,      U_i = A_n ... A_{i+1} z_i

    where ``E_i`` is the gap recorded when entry ``i`` was first applied. Every other recorded
    ``y`` is kept as is, so this removes only the entry's own effect; effects that flowed
    through later teacher answers (the "leak") need `remove_source`.

    Args:
        w_final: The live matrix after all ``entries``.
        entries: All ledger entries, in order.
        gaps: The gaps recorded for ``entries`` (`Replay.gaps`).
        i: Index of the entry to delete.
    """
    u = propagate_address(entries, i)
    result: FloatArray = w_final - entries[i].beta * np.outer(gaps[i], u)
    return result


def delete_entry_by_replay(w0: FloatArray, entries: Sequence[Entry], i: int) -> FloatArray:
    """Reference answer: replay every entry except ``i``, keeping the other ``y`` values."""
    kept = [entry for k, entry in enumerate(entries) if k != i]
    return replay(w0, kept).weights


@dataclass(frozen=True, eq=False)
class SourceRemoval:
    """Outcome of removing one source.

    Attributes:
        entries: Surviving entries, with recomputed teacher answers.
        weights: Matrix after replaying ``entries``.
        removed: Indices (in the original ledger) of deleted entries.
        recomputed: Indices (in the original ledger) of surviving entries whose ``y`` changed.
    """

    entries: list[Entry]
    weights: FloatArray
    removed: list[int]
    recomputed: list[int]


def remove_source(
    w0: FloatArray,
    entries: Sequence[Entry],
    source: str,
    teacher_without_source: Callable[[Entry], FloatArray],
) -> SourceRemoval:
    """Remove ``source`` from the ledger and rebuild the matrix.

    Source removal is the only operation that may recompute or delete ledger entries. It
    deletes every entry from the source, recomputes the recorded teacher answers of later
    entries with the source hidden (the "leak"), and replays the survivors. The result equals
    a run that never saw the source.

    Args:
        w0: Matrix at the start of the ledger (or a checkpoint before the first affected
            entry, with ``entries`` starting after it).
        entries: All ledger entries, in order.
        source: The source to remove.
        teacher_without_source: Returns the teacher answer for an entry when the source is
            hidden from the teacher's view. Hiding means masking the source's tokens from
            attention, never deleting them from the stream. It is called for every surviving
            entry after the first removed one; it must return the entry's current ``y``
            for entries the source never influenced.

    Replay always starts from ``w0``. Starting from the nearest checkpoint before the first
    affected entry gives the same matrix and is a pure optimisation for the real code.
    """
    removed = [k for k, entry in enumerate(entries) if entry.source == source]
    first = removed[0] if removed else len(entries)
    removed_set = set(removed)

    survivors: list[Entry] = []
    recomputed: list[int] = []
    for k, entry in enumerate(entries):
        if k in removed_set:
            continue
        if k > first:
            new_y = teacher_without_source(entry)
            if not np.array_equal(new_y, entry.y):
                recomputed.append(k)
                entry = entry.with_y(new_y)
        survivors.append(entry)

    return SourceRemoval(
        entries=survivors,
        weights=replay(w0, survivors).weights,
        removed=removed,
        recomputed=recomputed,
    )
