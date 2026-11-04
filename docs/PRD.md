# PRD: STARK

> What we are building, for whom, and how we know it worked. The why is in [research.md](research.md); the how is in [ARCHITECTURE.md](ARCHITECTURE.md). Changes to this file after the architecture is agreed should be deliberate, and noted in the change log at the end.

## Goal

STARK is a text-in, text-out assistant for bank compliance teams. It learns while it reads, records every update in a lesson ledger, and uses the ledger to:

- trace any answer to the documents that shaped it;
- remove any document or customer's data exactly, so the result equals an assistant that never read it.

It is also the platform for the project's three research questions:

- **RQ1, persistence:** how strongly and for how long an injected instruction persists under test-time learning.
- **RQ2, tracing:** whether answers can be traced exactly to the inputs behind them.
- **RQ3, removal:** whether removal is exact and much cheaper than a rerun.

There are two deliverables:

1. a working assistant with a ledger dashboard;
2. the measurements, an open benchmark built from the public RBI circular record, and a research paper.

## Users

| User | What they do with STARK | Features |
| --- | --- | --- |
| Branch staff | Ask compliance questions and get answers | Ask |
| Compliance officer | Feed in circulars, policies and emails; remove withdrawn circulars and fake notices | Read, Trace, Suppress, Erase, Verify |
| Data protection officer | Erase everything learned from one customer's records, with a report | Erase, Verify |
| IT security | Find every answer a forged input shaped, and remove the input | Trace, Suppress, Erase |
| Internal audit, RBI inspector | See why an answer was given and what was removed, when and by whom | Trace, Audit log |
| Researchers (the team, then paper readers) | Run the RQ1 to RQ3 experiments and the baselines on the benchmark | Benchmark and experiments |

In v1, STARK runs on one machine for one user at a time. Every action that changes what STARK knows records who did it, their role and the reason. There is no login system (see Out of scope).

## Core features

Each feature lists its acceptance statements. These are the test plan; slices will refer to them by ID.

### F1. Read and learn

STARK reads a stream of documents in date order: RBI circulars and Master Directions, internal policies, branch circulars, complaints, emails and customer records. One matrix in the model learns from what it reads, and every update is written to the ledger with the document and sources it came from.

- **F1.1** Every update appears in the ledger as one entry. The entry records:
  - its position in the stream;
  - its document;
  - the sources in its student view and in its teacher view;
  - its recorded numbers (addresses Z, teacher answers Y, step size β).
- **F1.2** Replaying the stored ledger from the pretrained weights gives the live learning layer **bit for bit** (same GPU model and software versions). This is Gate 1(a).
- **F1.3** Reading never changes any weight except the one learning matrix.
- **F1.4** On questions about documents that have left the context window, the learning assistant scores higher than the same assistant without learning (context-only). This is Gate 1(b).

### F2. Ask

Branch staff ask a question in plain text, and STARK answers in plain text.

- **F2.1** Asking never changes the learning layer or the ledger.
- **F2.2** Every answer is stored with the ledger version it was given under, so it can be traced later.
- **F2.3** A suppressed or erased source is never visible in the context an answer is given from.

### F3. Trace

For any answer, STARK shows which sources shaped it and by how much, down to individual ledger entries.

- **F3.1** At the learning layer, the per-entry contributions add up to the learned part of the answer, exactly up to float32 rounding.
- **F3.2** Sources are ranked by a score of how much they moved the answer. For the top sources, the score is checked by an ablation, as described in the architecture.
- **F3.3** On planted sources, the trace ranks the true source first in at least 90% of cases. This target is proposed and will be fixed after a pilot (RQ2).
- **F3.4** A trace can be exported as a report: question, answer, ledger version, and the sources with their scores.

### F4. Suppress a source (can be undone)

A compliance officer hides a source: a whole document, or a named set of spans such as one customer's details across many documents. STARK then behaves exactly as if it had read the stream with that source hidden. A suppressed source can be restored.

- **F4.1** After suppression, the learning layer equals that of a full rerun of the stream with the source hidden. The target is bit-identical; the accepted tolerance is fixed at Gate 2.
- **F4.2** Entries learned from the source are deleted. Entries learned while the source was only in the teacher's view (the leak) are recomputed with the source hidden. All other entries are reapplied unchanged.
- **F4.3** Removal costs much less than a full rerun, and the gap grows as sessions grow.
- **F4.4** Lessons from unrelated sources are kept: accuracy on questions about clean sources does not drop significantly (proposed).
- **F4.5** Restoring a suppressed source gives the same learning layer as if it had never been suppressed.

### F5. Erase a source (cannot be undone)

For a DPDP erasure request or a forged document, the data protection officer or an admin erases a source. Erase does everything suppress does, and also deletes the source's text and every stored number derived from it.

- **F5.1** After an erase, nothing under the run's storage folder contains:
  - the source's text;
  - its recorded addresses;
  - any pre-removal version of the entries or checkpoints it affected;
  - any stored answer or trace that cites it.
- **F5.2** The audit log keeps only a record that the erase happened. It holds the source's id and kind, the date, who erased it and why, and the verification result.
- **F5.3** An erased source can never be restored.
- **F5.4** Membership and extraction tests on erased customer records do no better than chance (proposed).

### F6. Verify

Every removal is checked, and the check is reported.

- **F6.1** A spot check runs after every suppress, restore and erase. It re-encodes a sample of entries the removal treated as unaffected, including those just outside the affected range, and confirms that their recorded numbers match bit for bit.
- **F6.2** A full check is available on demand and is used in every RQ3 experiment. It reruns the whole stream with the source hidden and compares the learning layers and the probe answers.
- **F6.3** Each check produces a report. It records:
  - the entries deleted, recomputed and reapplied;
  - the time taken;
  - the largest difference from the reference;
  - pass or fail.

### F7. Ledger and audit log

- **F7.1** The ledger can be listed by time, document and source, showing which entries each source touched and in which role (own entry, student view, teacher view).
- **F7.2** Every read, suppress, restore, erase and verification is an event in an append-only audit log with time, actor, role and reason.
- **F7.3** Suppress, restore and erase require a role allowed to do them. Erase is limited to the data protection officer and admin roles.

### F8. Dashboard (Phase 5)

A web interface (Streamlit) for compliance staff, as in the mock-up in the documentation (Figure 10). It has four screens:

- **Ask:** questions and answers with their traces.
- **Ledger:** documents and entries over time.
- **Removals:** with verification results.
- **Audit report.**

- **F8.1** The dashboard uses only STARK's Python interface. It never reads the ledger files directly.

### F9. Benchmark and experiments

- **F9.1** The benchmark is a stream of about 200 documents and about 100,000 tokens, against an 8,000-token window. It contains:
  - public RBI circulars and Master Directions in date order;
  - made-up internal policies, branch circulars, complaints, emails and customer records for a fictional co-operative bank;
  - planted fake notices, clearly marked as test data in their metadata.
- **F9.2** Probe questions are held out from the stream. Each probe has a correct answer, a wrong or planted answer where relevant, and the sources that determine it (the ground truth for tracing).
- **F9.3** Experiment runners exist for:
  - RQ1: persistence for a context-only agent and a learning agent, varying how often, where and how an instruction is injected;
  - RQ2: trace accuracy on planted sources;
  - RQ3: exactness, cost and knowledge kept after removing single entries and whole sources.
- **F9.4** Baselines are run on the same stream:
  - context only;
  - RAG over the same documents, deleting from the index;
  - full reset;
  - rerun without the source;
  - deleting only the source's own entries;
  - approximate unlearning by gradient ascent.

## Phasing

| Phase | Months | Features delivered | Gate or deliverable |
| --- | --- | --- | --- |
| 1 Learner and ledger | 1 to 2 | F1, F2, F7 (ledger part), first benchmark slice for F9; F3 to F6 prototyped on the toy example | **Gate 1:** replay rebuilds the live weights bit for bit; learning beats context-only on questions beyond the window |
| 2 Trace and remove | 3 to 5 | F3, F4, F5, F6, F7; benchmark complete | **Gate 2:** source removal matches a rerun within the agreed tolerance; knowledge from clean sources is kept |
| 3 Attacks and baselines | 6 to 9 | F9.3 (RQ1), F9.4 | **Gate 3:** STARK beats approximate unlearning on completeness, and full reset on knowledge kept |
| 4 Beyond the exact case | 10 to 12 | Two or three learning layers (measured approximation); a tool-using agent on emails and web pages | Report on how far removal drifts from a rerun |
| 5 Dashboard, paper, release | 13 to 15 | F8; benchmark and code release | Paper of 8 to 10 pages submitted; code and benchmark released |

If a gate fails, the result is reported and the plan changes before larger work starts. Capstone II covers months 1 to 8 and Capstone III months 9 to 15.

## Success criteria

The gates above, plus these measures. The first four are the standard measures of unlearning [Xu et al.]. Targets marked "proposed" are fixed after a short pilot.

| Measure | How it is measured | Target |
| --- | --- | --- |
| Erasure completeness | Distance between the learning layer after removal and a rerun without the source; difference in their outputs | Bit-identical, or within the tolerance agreed at Gate 2 |
| Time | Seconds to remove a source, against rerunning the session | Far below a rerun, and falling relative to it as sessions grow |
| Storage and memory | Ledger size and added time per update | Reported; about 1.5 MiB per entry, plus 64 MiB per checkpoint every 100 entries |
| Privacy | Membership and extraction tests on erased personal records | No better than chance (proposed) |
| Persistence of injection | Success rate of a planted instruction after it leaves the context window | Reported as the RQ1 result |
| Attack removed | Success rate of the planted instruction after removal | Same as a run that never saw it |
| Knowledge kept | Accuracy on questions about clean sources after removal | No significant drop (proposed) |
| Trace accuracy | How often the trace ranks the true source first | 90% or more (proposed) |
| Learning helps | Accuracy on questions about text beyond the context window, against context-only | Better than context-only |
| Answer quality against RAG | Accuracy on probe questions, including ones that need several circulars, against RAG over the same documents | Reported; the aim is parity or better |

## Out of scope

- **Media:** images, audio and video. STARK is text only.
- **Real deployment:** no deployment inside a real bank, no real customer data, and no legal certification of erasure. In particular, erase deletes files under the run's storage folder. It does not guarantee physical erasure from disks, backups or copies made outside STARK.
- **Models:** closed models reached only through an API, and models above about 2B parameters. The 8B repeat in the documentation is out of reach on the team's hardware.
- **Pretrained weights:** STARK never changes the model's pretrained weights, and never removes data from pretraining.
- **Exactness with several learning layers:** two or three layers are a measured approximation in Phase 4, not an exact guarantee.
- **Agent actions:** acting on what it reads (tool use) arrives only in Phase 4. Removal then fixes the memory, not actions already taken.
- **Attacks on the frozen model itself:** for example, tampering with its files.
- **Accounts and access control:** no login, user accounts, multi-user server or concurrent writers. Roles are recorded with each action and checked against a configured list.
- **Languages:** English only (proposed). The RBI publishes in English and Hindi; Hindi is not handled in v1.
- **Advice:** STARK is a research prototype. Its answers are not compliance or legal advice.

## Constraints

- **Hardware:**
  - one NVIDIA laptop (RTX 3050, 6 GB) for GPU work;
  - a workplace GPU that can be requested (its memory is to be confirmed with `nvidia-smi`);
  - two Macs and one Windows laptop for CPU work.
- **Public repository** under Apache 2.0. Nothing committed may contain secrets, weights, data, ledgers, real personal data, student ID numbers or RBI-style branding (see [AGENTS.md](../AGENTS.md)).
- **Timeline:** 15 months, with go or no-go gates at the end of Phases 1 to 3.

## Open questions

These are settled by measurement in Phase 1 and recorded in the architecture when they are:

1. Which MLP block hosts the learning layer.
2. The step-size constant η.
3. Whether answers read with the recent context window or with the question alone.
4. The tolerance for removal when exactness after re-encoding is not bit-identical on our GPUs.
5. Whether learning beats RAG on RBI questions. This is a Phase 3 result, and the project's top risk.

## Change log

| Date | Change |
| --- | --- |
| 29-10-2026 | First version, from the approved project documentation and the architecture discussion. Adds suppress (can be undone) and erase (cannot) as separate features; sources can be sets of spans, not only whole documents. |
