# AGENTS.md

Rules for everyone working in this repo, people and coding agents alike. Claude Code reads this through `CLAUDE.md`; the Antigravity CLI (`agy`) reads it directly. Keep it the single copy.

## Project
STARK ("Self-Taught, Auditable and Reversible Knowledge for Language Agents") is a PES University capstone: a text-in/text-out language agent on a frozen ~1B model (Llama 3.2 1B). Exactly one matrix (`down_proj` of one MLP block) learns during use through a teacher/student update rule (TTCD). Every update is logged in a lesson ledger, so we can replay the weights exactly, trace any answer to the entries and sources behind it, and remove a source exactly, including its leak through the teacher's long view.

The repo is public and Apache-2.0: everything committed is publishable. `docs/PRD.md`, `docs/ARCHITECTURE.md` and `docs/research.md` are written by the team; never invent their content.

## What exists
- `src/stark/reference.py`: the float64 numpy reference for the update rule, replay, backward trace, one-step entry deletion and source removal. The real (torch) code will be tested against it.
- `tests/test_toy.py`: the hand-checked 2x3 toy example. `tests/test_properties.py`: the same guarantees on random matrices.

Nothing else is built yet. Add a file or folder only when it has real content, and split a module into a package only when it outgrows one file.

## Commands
```
uv sync                              # install
uv run pre-commit install            # once per clone: checks every commit and commit message
uv run pre-commit run --all-files    # hygiene, ruff, gitleaks, blocked paths, student IDs
uv run pytest                        # tests, CPU only
```
Run the last two before saying work is done and before opening a PR. CI runs the same two on every PR, and also checks the PR's commit messages, title and description for agent attribution.

## Conventions
- Python 3.10+ (no newer syntax), `uv`, ruff with line length 100.
- First line of every new Python file: `# SPDX-License-Identifier: Apache-2.0`.
- Work on a branch and open a PR; a teammate other than the author approves it. Commits follow Conventional Commits (`feat(ledger): ...`).
- Agents: one agent per branch or worktree. Do not push, merge or add remotes unless asked.
- Only do the task you were given: no extras, no unrelated refactors. If something is ambiguous, stop and ask.
- `.devyard/` is each developer's own local scratch folder (gitignored). Agents do not open it unless asked.

## Invariants
Each one is load-bearing for the exact-removal proof. NEVER change code in a way that breaks one.

- **Numerics.** The real learning layer runs in float32 with deterministic kernels. Ledger tensors Z, Y and E are stored in 16-bit; checkpoints of W are float32. float64 is used ONLY in `reference.py` and the tests.
- **Addresses and teacher answers.** Addresses Z never depend on W (they come from frozen layers below the learning layer). Teacher answers Y are recorded constants computed with W as at session start; TTCD's current-weights teacher is an optional, separately flagged variant.
- **Append-only ledger.** During normal learning the ledger is append-only. Source removal is the ONLY operation that may recompute or delete entries, and every removal is authorised, logged and recorded as an event.
- **Hiding a source** means masking it from attention so later tokens keep their original positions. Never delete its tokens from the stream.
- **Constants.** Chunk size (64 tokens) and checkpoint interval (100 entries) are defined once, in `src/stark/config.py`, when the code that uses them is written. Never hard-code them anywhere else.
- **Exactness tests.** NEVER edit expected values in `tests/test_toy.py` or `tests/test_properties.py` to make a test pass. If one fails, report it and stop. The real gates (replay equals live; source removal equals a re-run) come with the learner and run manually on a GPU; CI is CPU only.

## Never
- Credit a coding agent in git or on GitHub. No `Co-Authored-By` trailer for an agent, no "Generated with ..." line and no other note naming an AI tool as author or contributor, in commit messages, PR titles or PR descriptions. Commits and PRs belong to the person who makes them. The commit-msg hook and CI reject such lines; never bypass them with `--no-verify`.
- Read or write `.env*`. Secrets live in each developer's local `.env`; the variable names are `HF_TOKEN` and `WANDB_API_KEY`. Ask the human for values.
- Use `git add -f`, rewrite history, or edit `LICENSE`.
- Commit secrets, weights, data, ledgers, checkpoints, experiment outputs, student ID numbers, real personal data, PDFs or RBI-style branding (letterheads, logos, official imagery). The pre-commit hooks block most of these.
- Make network calls or download weights in tests.
- Invent benchmark data, RBI-style notices or customer records.
