# STARK: project context

Last updated: 28 October 2026.

This file carries the full history and current state of the STARK capstone project from a previous chat. Paste it at the start of a new chat. The next piece of work is the **architecture design** of STARK, so the design spec (section 5) and the open questions (section 13) matter most.

## How to use this file (for the assistant)

- Treat everything here as settled unless the user says otherwise. Sections 5 to 9 come from the project documentation the team submitted.
- The next task is turning the design into a buildable architecture: modules, data flow, storage, interfaces and the order of building. Start from section 13.
- The user writes and reads British English. Use plain words, no buzzwords, no em dashes, and lead with the most important point.
- The user wants depth: from plain-language framing down to implementation detail and feasibility on their real hardware (section 8).

---

## 1. Where we stand today

- **Project:** STARK, Self-Taught, Auditable and Reversible Knowledge for Language Agents.
- **Status:** approved by the faculty on 9 October 2026. The design is complete and no code has been written.
- **Done since approval:**
  - developer documentation;
  - literature search: about 80 papers, with 12 picked as the core set;
  - tech stack finalised (23 October);
  - GitHub repo created, with a README.
- **Next:**
  - architecture design (this new chat);
  - then a requirements specification;
  - then setting up the development environment.
- **Big change since the documentation:** the hardware is smaller than planned. The documentation assumed a 24 GB GPU. The team has 6 GB laptop GPUs, plus a workplace GPU they can request (section 8).

## 2. Project at a glance

| Item | Detail |
| --- | --- |
| Course | PES University, UE24CS320A Capstone Project Phase I |
| Team | Batch 39, four third-year students: Mitesh Kurumeti, Mohammed Tahir Siddique, Pavan Kishor M, Riya Harugop |
| Guide | Dr. Geetha Dayalan |
| Timeline | About 15 months of build, over Capstone II (months 1 to 8) and Capstone III (months 9 to 15) |
| First users | Compliance teams at RBI-regulated co-operative banks and NBFCs |
| Repo | On GitHub, Apache 2.0 licence, README written (team left out of the README on purpose) |
| Repo description | STARK is an auditable language agent that learns while it works, records every update in a lesson ledger, and helps bank compliance teams control what it knows by tracing any answer to its sources, removing any source exactly, and checking each removal against a clean rerun, all without retraining the model. |

## 3. How the project came about

| When | What happened |
| --- | --- |
| 10 to 21 Aug 2026 | **KYBER** was the first and only idea before STARK: a self-healing Linux OS using reinforcement learning (PPO) over eBPF telemetry. The team wrote a proposal and roadmap, then ran feasibility checks. |
| 24 Aug | **KYBER dropped as not technically feasible** (reasons in section 4). |
| 25 to 28 Aug | New idea: an AI agent that keeps learning while it works (test-time learning), tested on the ARC-AGI-3 interactive game benchmark. **Old STARK** chosen. |
| 31 Aug to 11 Sep | 12 papers split, 3 per member; each member wrote a findings document; team discussion. Found that Test-Time Context Distillation (TTCD, Wang et al., arXiv 2608.01672) already covered the core learning objective, so the contribution was narrowed to two hypotheses: H1 (hindsight write) and H2 (erase-before-write correction). |
| 14 to 18 Sep | Old STARK proposal written: In-Place TTT plus TTCD on a frozen 1B Llama model, a rule-reversal toy experiment (8x8 grid game, 60-turn sessions, conditions C0 to C5) and decision gates. Discussed with the guide on 17 Sep. |
| 21 to 26 Sep | Internal exams; no project work. |
| 28 Sep to 1 Oct | Review 1 preparation: documentation, an 18-slide PPT, two mock presentations. |
| **6 Oct** | **Review 1.** The panel was not convinced: the end product was unclear, it looked like existing papers put together for a game-playing agent, and they asked how the weights actually get updated. The same day they asked for a specific niche and its users, not "everyone". |
| 7 Oct | **Project changed** to the current STARK (section 4). |
| 7 to 8 Oct | Niche chosen: compliance at RBI-regulated co-operative banks and NBFCs. New PPT made in the university template. 29-page project documentation and a one-page summary written. |
| **9 Oct** | **Project approved by the faculty.** Requirements gathering began. |
| 14 Oct | The panel asked for a brief on top risks and a fallback plan. Sent informally to the guide (section 10). |
| 12 to 16 Oct | Developer documentation written up. |
| 19 to 23 Oct | Literature search on IEEE Xplore, ACM Digital Library and arXiv (about 80 papers in 9 areas). 12 core papers picked. Tech stack finalised. |
| 22 Oct | GitHub repo created; README drafted. |

Weekly status reports in the university template exist for 10 Aug to 2 Oct and for 5 Oct to 23 Oct.

## 4. Decision log

| Decision | Why |
| --- | --- |
| Dropped KYBER | (1) RL on a live kernel runs in real time at one step per second: 1 million PPO steps take about 11.5 days on one VM, and the planned 9-configuration grid search needed over 3 months. (2) eBPF needs root and matching kernel headers, and the VMs need nested virtualisation; neither is available on shared lab machines. (3) There is no public dataset of real compound OS faults, only synthetic ones. (4) Reward is read one second after an action, which misses slow faults, and kill or restart actions are unsafe on shared machines. |
| Dropped old STARK's framing | TTCD already covered its core objective, and the panel saw it as a game agent built from existing papers, with no clear end product. |
| Dropped H1 and H2 hypotheses and the game setting | Replaced by a mechanism with a clear product: audit which update an answer came from, and remove a specific update or source exactly. |
| Text-in, text-out language agent | Clear end product and real users; no game environment needed. |
| Niche: compliance at RBI-regulated co-operative banks and NBFCs | The RBI rulebook keeps changing and withdraws rules; smaller lenders get most RBI penalties; fake RBI letterheads are a known fraud; India's data protection law adds erasure duties. Each of these maps directly to trace and remove. |
| One learning layer only | Keeps removal exact (section 5.8). Two or three layers are a measured extension in Phase 4. |
| Teacher reads with the session-start weights | The teacher's answers become fixed numbers that can be recorded, so traces add up exactly. TTCD's original teacher, which uses the current weights, is tested as a variant. |

---

## 5. STARK design (full spec)

### 5.1 Problem statement

Compliance teams at RBI-regulated co-operative banks and NBFCs must keep up with a rulebook that keeps changing. Language-model assistants that update their own weights during use could keep up, but they keep no record of what each input taught them. A forged, private or withdrawn document can therefore become a lasting part of the assistant's behaviour that can be neither traced nor removed without discarding all learning.

STARK is a text-to-text assistant whose test-time learning is fully logged, so that (a) any answer can be traced to the inputs that shaped it, and (b) the effect of any input can be removed exactly. The project also measures how far test-time learning, without such controls, turns an injected instruction into lasting behaviour.

### 5.2 Users and use cases

| Situation | Who acts | Without STARK | With STARK |
| --- | --- | --- | --- |
| An RBI circular is withdrawn or replaced | Compliance officer | The assistant keeps applying the old rule; the only fix is to wipe all learning | Removes the circular exactly; the trace confirms no answer still depends on it |
| A fake RBI notice got in | Compliance and IT security | Nobody can tell which answers it shaped | The trace finds every answer it shaped; STARK removes it, including its leak into later lessons |
| A customer asks for erasure under the DPDP Act | Data protection officer | Data can't be removed without wiping all learning | Removes every lesson learned from that customer's records, with a verification report |
| An auditor or RBI inspector asks why | Internal audit | No record of where an answer came from | A trace report: which documents produced the answer, and by how much |

Supporting facts used in the documentation:
- The RBI withdrew 9,445 circulars on 28 Nov 2025 and folded them into 244 Master Directions.
- Of 353 RBI penalties in 2024-25, 264 were on co-operative banks and 37 on NBFCs and ARCs.
- The RBI's FREE-AI framework (Aug 2025) recommends AI audits and deletion.
- DPDP erasure duties apply from May 2027.
- The RBI warns of fraud on fake letterheads.

End product: the assistant plus a **ledger dashboard**. Branch staff ask questions as normal. An auditor opens any answer and sees which documents shaped it. An admin removes a document and sees the result checked against a fresh run that never read it.

### 5.3 Research questions

- **RQ1, persistence.** When a language model learns during use, how strongly and for how long does an injected instruction keep affecting its answers after it has left the context window?
- **RQ2, tracing.** Can every answer be traced, exactly, to the updates and inputs that produced it?
- **RQ3, removal.** Can the effect of any input be removed exactly, including its indirect influence on later updates, at a cost far below rerunning the session?

**Exact removal** means: after removing source S, the learning layer equals, up to floating-point rounding, the learning layer of a run that read the same stream with S hidden. This follows SISA's definition of exact unlearning.

### 5.4 The learning mechanism

- The backbone is frozen: Llama 3.2 1B, open weights. Only one matrix learns: `down_proj` (W) at the end of one MLP block. For Llama 3.2 1B, the MLP expands a 2,048-number vector into an 8,192-number vector z, and W (2,048 x 8,192) maps it back: `y = W z`.
- Read z as an **address** and y as the **content** stored there:
  - Write: `W ← W + v kᵀ`.
  - Read at k: `W k = v ‖k‖²`.
  - Orthogonal addresses don't interfere; the same address collides.
- **Key fact:** the address z never depends on W. It is computed before W is applied, by frozen layers below.
- **Update rule (TTCD).** The same model reads the same text twice. The **student** sees only the last few tokens; the **teacher** sees much further back (proposed 8,000-token window). At token t, the student's address is `z_t`, its output is `W z_t`, and the teacher's output is `y_t`. The gap is `e_t = y_t − W z_t`.
- For one chunk C of about 64 tokens with step size β:

```
W ← W + β Σ_t (y_t − W z_t) z_tᵀ                      (2)
  = W (I − β Σ_t z_t z_tᵀ) + β Σ_t y_t z_tᵀ             (3)  "erase at these addresses, then write the teacher's answer"
```

The update runs once per chunk, and the new W is used from the next chunk on.

### 5.5 Components

| Component | What it does |
| --- | --- |
| Frozen language model | Open model of about 1B parameters (Llama 3.2 1B or similar). Weights never change. |
| Learning layer | `down_proj` of one MLP block, updated once per chunk with the TTCD rule. |
| Teacher and student views | The same model run with a long and a short attention window. |
| Lesson ledger | One entry per update, plus a saved copy of W every 100 entries. |
| Tracer | Works out each entry's share of an answer and groups the shares by source. |
| Remover | Removes one entry or a whole source and replays the ledger. |
| Verifier | Checks a removal against a rerun that never saw the source. |
| Dashboard | Entries over time, the trace of any answer, and controls for removal. |

### 5.6 Design settings

| Setting | Value | Reason |
| --- | --- | --- |
| Where learning happens | One `down_proj` in one block | The layer's inputs come only from frozen layers, so every update is a fixed operation that can be replayed exactly |
| Teacher's weights | Session-start W | Teacher answers are fixed numbers that can be recorded; TTCD's original (current-weights) teacher tested as a variant |
| Chunk size | 64 tokens | One ledger entry per chunk |
| Checkpoints | A copy of W every 100 entries | Removal replays from the nearest checkpoint, not from the start |
| Teacher window | Proposed 8,000 tokens | Long view |
| Precision | 32-bit arithmetic for the learning layer; deterministic kernels | Exactness checks |

### 5.7 Ledger entry format (1B model, 64 tokens, 16-bit numbers)

| Field | Contents | Size |
| --- | --- | --- |
| Entry number and time | Order of the update | a few bytes |
| Sources in view | Which documents were inside the student's window and inside the teacher's window | a few bytes |
| Addresses Z | Student's input to the learning layer, one per token | 8,192 x 64, 1 MiB |
| Teacher's answers Y | Teacher's output, one per token | 2,048 x 64, 0.25 MiB |
| Step size β | How strongly the entry was written | one number |
| Gap E | Teacher's answer minus the student's output at write time | 2,048 x 64, 0.25 MiB |

The total is about 1.5 MiB per entry, or 1.5 GiB per 1,000 entries (64,000 tokens). A 100,000-token run gives a ledger of about 2.3 GB. Each saved copy of W (32-bit) is 64 MiB.

Planned storage: memory-mapped files for the numbers, with an SQLite index of entries and sources.

### 5.8 Replay, trace and removal

**Replay.** Each entry is a fixed operation that uses only recorded numbers:

```
W_i = W_{i-1} (I − β_i Z_i Z_iᵀ) + β_i Y_i Z_iᵀ        (4)
```

Because Z and Y are recorded and don't depend on W, replaying the same entries from the same start always gives the same W. The ledger is the source of truth, and W is computed from it, like a bank balance computed from the opening balance and the list of transactions.

**Trace.** A question reaches the learning layer at address q. Carry q backwards through the ledger:

```
r_n = q,   r_{i-1} = r_i − β_i Z_i (Z_iᵀ r_i),   share_i = β_i Y_i (Z_iᵀ r_i)      (5)
```

The shares, plus what is left of the starting weights, add up exactly to `W q`. Summing shares by source gives the trace. The cost is about 2 million operations per entry, so tracing over 1,000 entries takes milliseconds.

**Remove one entry.** Replay without it, or use the one-step shortcut:

```
W ← W − β_i E_i U_iᵀ,   U_i = A_n ⋯ A_{i+1} Z_i,   A_j = I − β_j Z_j Z_jᵀ      (6)
```

Both give the same W. This was checked in code on random matrices, where they agree to within 1e-15.

**Remove a whole source.** A source touches two kinds of entry:
- entries written while the student was reading it;
- entries written soon after, while it was still inside the teacher's view. This is the **leak**.

The four steps:
1. Find every entry whose student or teacher view held the source.
2. Recompute those entries with the source hidden from attention, and drop the source's own tokens.
3. Replay from the last checkpoint before the first affected entry. Entries after the teacher window are reapplied exactly as recorded.
4. Delete the source's recorded addresses from the ledger, since they were computed from its text.

Only entries within one teacher window of the source need the model. Later ones are cheap matrix arithmetic.

**Why removal is exact.** Recorded inputs don't change, because addresses come from frozen layers. And each update is a fixed operation. So replaying without a source gives exactly the learning layer of a run that never saw it.

| Condition for exactness | Holds when | If not |
| --- | --- | --- |
| Addresses don't depend on W | One learning layer | Several layers: removal becomes approximate; gap measured against a full rerun |
| What is read doesn't depend on the agent | Agent reads a stream of documents | If the agent acts, removal fixes the memory, not past actions; gap measured |
| Hiding a source equals deleting it | Always, up to token positions | Hiding keeps later tokens at original positions; effect expected small, measured |

### 5.9 Worked example (in the documentation, checked in code)

Toy setting: 3-number addresses, 2-number contents, β = 0.5, teacher view of 4 sentences, student view of 1.

| Sentence | Source | Address z | Teacher answer y |
| --- | --- | --- | --- |
| S1 | RBI KYC rule: high-risk customers need re-KYC every two years | (1, 0, 0) | (+1, 0) |
| S2 | Internal rule: cash deposits above ₹50,000 need PAN | (0, 1, 0) | (0, +1) |
| S3 | Fake RBI letterhead email: re-KYC suspended | (1, 0, 0) | (−1, 0) |
| S4 | Branch circular: follow the most recent RBI instruction | (0.6, 0, 0.8) | (−0.6, 0) with the notice in view; (+0.6, 0) with it hidden |

Results:
- After all four sentences, Q1 ("is re-KYC due?") reads −0.385 (not due) and Q2 reads −0.375.
- The trace of Q1:

  | Entry | Share |
  | --- | --- |
  | E4 | −0.180 |
  | E3 (fake notice) | −0.410 |
  | E2 | 0 |
  | E1 | +0.205 |
  | Total | −0.385, exactly the answer |

- Deleting only E3 gives Q1 +0.23, but Q2 is still −0.15: the notice has leaked through E4.
- STARK's source removal gives Q1 +0.59 and Q2 +0.45. This is identical to a run that never saw the notice, and the PAN lesson is untouched.

---

## 6. Evaluation plan

**Data (RBI compliance stream).**
- Public RBI circulars and Master Directions in date order.
- Mixed in: made-up internal policies, branch circulars, customer complaints and emails for a fictional co-operative bank.
- Withdrawn circulars give real ground truth for removal.
- Fake notices on imitation letterheads, clearly marked as test data, are planted for RQ1.
- Made-up customer records are used for erasure.
- Proposed size: about 200 documents, around 100,000 tokens, against an 8,000-token window.
- Scoring uses held-out probe questions.
- No real customer data is used.
- IndiaFinBench (Pall, arXiv 2026) is kept as an RBI test set.

Phase 4 adds a tool-using agent that reads emails and web pages with hidden injected instructions.

**Experiments.**
- RQ1: vary how often an instruction is injected, where it goes, and its wording. Measure success over later documents, for a context-only agent and a learning agent.
- RQ2: plant known sources and check that the trace ranks the true source first.
- RQ3: remove single entries and whole sources. Compare with a rerun, measure the leak left by deleting only a source's own entries, and measure cost and the clean knowledge kept.

**Baselines.**

| Baseline | Exact removal? | Keeps other learning? |
| --- | --- | --- |
| Context only (no learning) | Yes | Yes |
| RAG over the same documents, delete from index | Yes | Yes (but the model learns nothing) |
| Learning, then full reset | Yes | No |
| Learning, then rerun without the source (reference) | Yes | Yes |
| Learning, then delete only the source's own entries | No (leak) | Yes |
| Learning, then approximate unlearning (gradient ascent) | No | Mostly |

**Measures and targets.**
- Erasure completeness: within rounding of a rerun.
- Time: far below a rerun.
- Storage: about 1.5 GiB per 1,000 entries.
- Privacy: membership and extraction tests no better than chance (proposed).
- Persistence of injection: the RQ1 result.
- Attack removed: same as a run that never saw it.
- Knowledge kept: no significant drop (proposed).
- Trace accuracy: true source ranked first 90% or more (proposed).

## 7. Tech stack (finalised 23 Oct 2026)

| Layer | Choice |
| --- | --- |
| Model | Llama 3.2 1B (open weights); a similar 1B to 2B open model as fallback |
| Framework | Python, PyTorch, Hugging Face Transformers |
| Learning layer | A hook on one MLP block's `down_proj`; teacher and student views through sliding-window attention masks |
| Ledger | Memory-mapped files for the numbers, with an SQLite index of entries and sources |
| Dashboard | Streamlit |
| Experiment tracking | Weights & Biases |

The old STARK's stack (Triton, Gymnasium, IndiaAI Compute Portal H100) is no longer the plan. Gymnasium was for the game setting.

## 8. Hardware: planned vs actual

**Planned in the documentation.**
- About 6 to 8 GB of GPU memory in use for the 1B model with an 8,000-token teacher window. A 24 GB card is comfortable; 16 GB is slower.
- About 5 to 10 minutes per 100,000-token run on an A100-class card, and about 100 GPU-hours for all 1B experiments. An optional 8B repeat needs 40 to 80 GB.
- 32 GB RAM and 200 GB disk.
- No backpropagation through the network: the GPU mainly runs the model forward.

**Actual (stated 28 Oct 2026).**
- Team laptops: RTX 3050, 6 GB.
- Workplace: an "RTX 4050, 12 GB" can be requested. **Confirm the memory with `nvidia-smi`**: as far as we know the RTX 4050 is a 6 GB laptop GPU, so this may be a different card (for example an RTX 3060 or 4070, both 12 GB).

**What this means for the architecture (to work through in the new chat).**
- Llama 3.2 1B weights in 16-bit are about 2.5 GB. Its key-value cache is about 32 KiB per token, about 256 MiB for 8,000 tokens. W in 32-bit is 64 MiB. So 12 GB should fit the 1B model with an 8,000-token teacher window; 6 GB is tight and probably needs shorter windows for development.
- Run times will be longer than the A100 estimates. Time one run early.
- The 8B repeat is out of reach on this hardware.
- Exactness checks (replay vs rerun) must run on the same GPU model. Don't compare a laptop 3050 run with a workplace-card run.

## 9. Roadmap and work split

| Phase | Months | Work | Gate or deliverable |
| --- | --- | --- | --- |
| 1 Learner and ledger | 1 to 2 | One learning layer with the TTCD update on Llama 3.2 1B; the ledger; replay | **Gate 1:** replay rebuilds the live weights within rounding; learning beats context-only memory on questions beyond the window |
| 2 Trace and remove | 3 to 5 | Tracing; removing entries and sources; the benchmark stream | **Gate 2:** source removal matches a rerun within rounding; knowledge from clean sources is kept |
| 3 Attacks and baselines | 6 to 9 | RQ1 persistence experiments; all baselines; the measures | **Gate 3:** STARK beats approximate unlearning on completeness and full reset on knowledge kept |
| 4 Beyond the exact case | 10 to 12 | Two or three learning layers (measured approximation); tool-using agent | Report on how far removal drifts from a rerun |
| 5 Dashboard, paper, release | 13 to 15 | Dashboard and demo; 8 to 10 page paper; code and benchmark release | Paper submitted; code and benchmark released |

If a gate fails, the result is reported and the plan changes before larger work starts.

| Member | Leads | Capstone II | Capstone III |
| --- | --- | --- | --- |
| Mitesh Kurumeti | Learning layer | TTCD update on Llama 3.2 1B; teacher and student views | Two-to-three-layer drift study |
| Mohammed Tahir Siddique | Ledger and tracing | Ledger storage, checkpoints, exact replay, tracing | Scaling and cost measurements |
| Pavan Kishor M | Removal and verification | Entry and source removal, the leak, checks against reruns | Baselines: RAG, approximate unlearning, full reset |
| Riya Harugop | Data, attacks and users | RBI circular stream, fake notices, made-up customer records | RQ1 persistence study; dashboard |

Each member also reviews another member's work every week.

## 10. Risks

**Sent to the guide on 14 Oct, as the panel asked.**

1. Top risks:
   - (a) Learning during use may not beat plain RAG on RBI compliance questions. If it doesn't, there's little reason to update the weights.
   - (b) Exact removal may be too slow, or not fully exact. Replay gets longer as sessions grow, and GPU results can differ slightly between runs.
2. Fallback plan:
   - (a) The ledger works either way, so tracing and removal still get delivered. Report the accuracy result as it is, and the ledger can move onto a LoRA adapter if needed.
   - (b) Checkpoints at fixed intervals keep replay short, and PyTorch runs in deterministic mode. If it's still too slow, subtract the logged update instead and measure what's left against exact removal.

**From the documentation.**
- **One layer may learn too little:** tested at Gate 1; add layers if needed and report the approximation.
- **Several layers break exactness:** start with one and measure the gap.
- **Agents that act:** removal fixes the memory, not past actions.
- **The ledger stores data derived from text:** removing a source deletes its recorded addresses.
- **Storage grows:** recompute teacher answers on demand, or use a retention window.
- **Removal requests can be abused:** they need authorisation and are logged.
- **Related work moves fast:** check monthly and release results early after Gate 2.

## 11. Related work

**12 core papers.**

| # | Paper | Venue | Role for STARK |
| --- | --- | --- | --- |
| 1 | Feng et al., In-Place Test-Time Training (arXiv 2604.06169) | ICLR 2026 | Which matrix learns and how |
| 2 | Wang et al., Learning What to Remember: Test-Time Training via Context Distillation (arXiv 2608.01672) | arXiv 2026 | STARK's update rule |
| 3 | Meng et al., ROME: Locating and Editing Factual Associations in GPT (arXiv 2202.05262) | NeurIPS 2022 | Rank-one write to an MLP matrix stores a fact, the same form as a ledger entry |
| 4 | Bourtoule et al., Machine Unlearning (SISA) | IEEE S&P 2021 | Exact unlearning, checkpoint and replay |
| 5 | Graves et al., Amnesiac Machine Learning (arXiv 2010.10981) | AAAI 2021 | Nearest earlier update ledger; its leftover influence is the leak STARK fixes |
| 6 | Unlearning at Scale: State-Exact Trace-Preserving Deletion in Billion-Parameter LMs (arXiv 2508.12220) | arXiv 2025 | Closest work: exact logged replay, but for fine-tuning data, not learning during use |
| 7 | Ramesh, Subtract, Transport, or Replay? Auditable Deletion from Language-Model Memory (arXiv 2607.27539) | arXiv 2026 | Closest work: exact deletion from in-context memory; weights never change |
| 8 | Pruthi et al., TracIn (arXiv 2002.08484) | NeurIPS 2020 | RQ2 baseline: approximate tracing over checkpoints |
| 9 | Greshake et al., Indirect Prompt Injection (arXiv 2302.12173) | ACM AISec 2023 | RQ1 threat model |
| 10 | Lewis et al., Retrieval-Augmented Generation (arXiv 2005.11401) | NeurIPS 2020 | Main baseline |
| 11 | Abualhaija et al., Automated QA for Compliance Requirements | IEEE RE 2022 | Evaluation design for regulation QA |
| 12 | Veale et al., Algorithms that Remember (arXiv 1807.04644) | Phil. Trans. R. Soc. A 2018 | A model can be personal data, so erasure must reach the weights |

Other key references: Titans (arXiv 2501.00663); Xu et al.'s unlearning survey (arXiv 2308.07061); and exact unlearning for ridge heads on frozen models (arXiv 2603.12977).

**What STARK adds.**
- The first measurement of injected instructions persisting through test-time learning in language models (RQ1).
- Exact tracing of answers to inputs, for test-time learning (RQ2).
- Exact removal for test-time learning, including the leak through the teacher's view, with recomputation limited to one teacher window (RQ3).
- A design under which the one-step transport shortcut is exact.
- A benchmark built from real RBI circular withdrawals.

## 12. Files produced so far

These live in the previous chat. Attach any you need to the new chat.

- `STARK_Project_Documentation.docx` / `.pdf`: the 29-page design document (source for sections 5 to 11).
- `STARK_One_Page.docx` / `.pdf`: a one-page summary.
- `STARK_Review_Deck.pptx`: the approval PPT in the university template.
- `STARK_12_Core_Papers.pdf`: the 12 papers with links. It is dated 9 Oct, while the weekly report says 22 Oct.
- STARK Related Research Papers: a document listing about 80 papers in 9 areas.
- Weekly reports: `STARK_Weekly_Reports_10Aug-02Oct_2026.docx` and `STARK_Weekly_Reports_05Oct-23Oct_2026.docx`.
- `README.md`: the repo README. It includes this planned layout:

```text
stark/
├── learner/      learning layer and TTCD update
├── ledger/       entries, checkpoints and replay
├── trace/        per-entry and per-source attribution
├── remove/       entry and source removal, verification
└── dashboard/    Streamlit interface
data/             RBI stream and synthetic documents
experiments/      RQ1 to RQ3 configurations and runs
docs/             design documentation
```

## 13. Open questions for the architecture design

These are not settled in the documentation and are the agenda for the new chat.

1. **Which block hosts the learning layer?** Llama 3.2 1B has 16 blocks.
2. **Window sizes.**
   - The student window is not fixed ("the last few tokens").
   - The teacher window is proposed at 8,000 tokens.
   - Chunk is 64 tokens.
   - What window is used when answering questions?
3. **Step size β:** fixed, decaying, or normalised by the size of Z? Stability over thousands of entries.
4. **Precompute or online?** Addresses come from frozen layers, and the teacher uses session-start weights, so Z and Y for a whole stream may be computable in one forward pass up to the learning block, with the W updates then done as pure matrix arithmetic. This follows from the design but needs confirming in Phase 1. If it holds, it shapes the whole pipeline and cuts GPU time.
5. **Answering.**
   - Answers must never write to W.
   - A multi-token answer touches the learning layer at many addresses. How is the trace aggregated over them?
6. **From shares to a source score.** Each share is a 2,048-number vector in the layer's output space. How does it become one number per source for RQ2 (for example, its effect on the answer's logits)?
7. **Hiding a source.** Use an attention mask that keeps token positions, and measure the position effect.
8. **Exactness rules.**
   - The live learner must consume exactly the stored (16-bit) Z and Y, so that live and replayed W match bit for bit.
   - The learning layer uses 32-bit arithmetic.
   - Use `torch.use_deterministic_algorithms`.
   - Replay and rerun on the same card.
9. **Storage layout.**
   - How memory-mapped arrays are laid out.
   - SQLite schema: entries, sources, which sources each entry saw and in which role (student or teacher), checkpoints.
   - Retention policy.
10. **Interfaces.**
    - Python API, for example `learn(doc)`, `ask(question)`, `trace(answer)`, `remove(source)`, `verify(source)`.
    - How the Streamlit dashboard reads the ledger.
11. **Hardware fit:** what runs on the 6 GB laptops (development, unit tests, short windows) and what needs the 12 GB card (full runs).
12. **Baselines:** the RAG stack (embedding model and index) and how approximate unlearning (gradient ascent) is run.
13. **Repo structure and build order** for Phase 1, aimed at Gate 1.

## 14. Working preferences

- British English, plain words, no buzzwords, no em dashes.
- Most important point first; no build-up.
- Ask before building when a choice is expensive to redo; otherwise make a sensible call and say so.
- Dates in weekly reports use the DD-MM-YYYY format, in Monday to Friday week blocks.
