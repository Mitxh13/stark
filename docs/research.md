# Research: STARK

> Why STARK exists, who it is for, and what already exists. Sources: the project documentation approved on 9 October 2026, and the team's literature search (19 to 23 October 2026: about 80 papers in 9 areas, 12 picked as the core set). The PRD builds on this file; the architecture builds on the PRD.

## The problem

Language models are starting to learn while they are used. A small part of their weights changes as they read, so they can keep knowledge that no longer fits in their context window. This learning keeps no record of what was learned from which input. A forged, private or withdrawn document can therefore become a lasting part of the model's behaviour, and nobody can trace it or take it out without wiping everything the model has learned.

**In plain words:** an assistant that learns on the job can be taught something false, such as a fake RBI notice, and today nobody can find out where it learned it or take it back out.

Consider a compliance assistant at a co-operative bank that learns on the job for a month. In that time it reads:

1. an email on a fake RBI letterhead saying that periodic re-KYC for high-risk customers is suspended;
2. a customer's complaint, and later that customer asks for their data to be erased;
3. an RBI circular that is later withdrawn and replaced.

Today the only way to undo any of these is to wipe everything the assistant learned that month. These are exactly the three reasons the unlearning literature gives for removing data from a model: security against poisoned data, privacy, and keeping a model current [5].

**Why learning during use makes this worse.** An instruction planted in a document normally stops affecting a model once the document leaves the context window. If the model learns from what it reads, the instruction can be written into its weights and keep acting after the document is gone. Nobody has measured how strongly or for how long this happens in language models (see "Attacks" below).

## Who has this problem

**Compliance teams at RBI-regulated co-operative banks and NBFCs.** They read every change to the rulebook, turn it into internal policy, answer questions from branch staff, and produce evidence for auditors and RBI inspectors.

| Role | What they need from an assistant that learns |
| --- | --- |
| Compliance officer | Apply the current rules; forget a withdrawn circular cleanly |
| Branch staff | Ask a question and get the current answer |
| Internal audit, RBI inspector | See why the assistant gave an answer, and from which documents |
| Data protection officer | Remove everything learned from one customer's records, with proof |
| IT security | Find every answer a forged input shaped, and remove the input |

Why smaller lenders first:

| Fact | What it means for an assistant that learns |
| --- | --- |
| The RBI withdrew 9,445 circulars on 28 November 2025 and folded them into 244 Master Directions [19] | The rulebook changes, and withdrawn rules must be forgotten cleanly |
| Of 353 RBI penalties in 2024-25, 264 were on co-operative banks and 37 on NBFCs and ARCs [20] | Smaller lenders, with small compliance teams, get the rules wrong most often |
| The FREE-AI framework has 26 recommendations, including AI audits, incident reporting and deletion [21] | Every answer must be explainable, and every lesson removable |
| DPDP erasure duties apply from May 2027 [22]; the RBI warns of fraud on fake letterheads [18] | Customer data and forged inputs must both be removable on request |

The RBI publishes every circular with its date of issue and of withdrawal. That gives the project a real stream of documents with real withdrawals, which serve as ground truth for testing removal.

## Why now

- **The learning methods are new.** Titans (2025) [3], In-Place Test-Time Training (ICLR 2026) [1] and Test-Time Context Distillation (2026) [2] made learning during use practical for standard transformers. All three aim at capability. None of them records what was learned from which input.
- **Regulation now expects it.**
  - The RBI's FREE-AI framework (August 2025) recommends AI audits and data governance that covers deletion [21].
  - India's DPDP Rules were notified in November 2025, and their erasure duties apply from May 2027 [22].
  - The EU AI Act requires automatic logs for high-risk AI systems, and the GDPR gives a right to erasure [17].
  - Governance researchers argue that systems which learn during deployment need their updates logged [14].
- **The rulebook just churned.** The November 2025 withdrawal of 9,445 circulars is a large, public, dated record of rules that must be forgotten.

## What exists already

### How compliance teams cope today

Teams read circulars by hand, turn them into internal policy, and answer branch questions themselves. An AI assistant today is either context-only (it forgets what scrolls out of its window) or retrieval-based (RAG). Neither learns from the stream, so neither has the problem STARK solves, and neither gets the benefit of learning.

### Research, by area

| Area | Work | What it gives STARK | What it leaves out |
| --- | --- | --- | --- |
| Learning during use | In-Place Test-Time Training [1] | `down_proj` can change during use in a standard transformer | No record of what was learned |
| | TTCD [2] | The teacher-student update rule STARK uses | No record; no removal |
| | Titans [3] | Learning in a separate memory module | No record; no removal |
| Editing facts in an MLP | ROME [24] | A rank-one write to an MLP matrix stores a fact, the same form as a ledger entry | Edits are deliberate, not learned from a stream |
| Exact unlearning | SISA [4] | The definition of exact unlearning; checkpoint and replay | Built for training data split into shards, not learning during use |
| | Xu et al. survey [5] | Exact vs approximate unlearning; the four standard measures | Names streaming data and transformers as open problems |
| Logged updates | Amnesiac Machine Learning [6] | Store each update and subtract it | Approximate, because later updates were built on the removed one. This is the leak STARK fixes |
| | Unlearning at Scale [7] | Log every step and replay with the data switched off; bit-exact on 1B to 3B language models | Fine-tuning data, not learning during use |
| Linear part on a frozen model | Ridge heads [8], mixed-privacy forgetting [9] | Keep the learnable part linear so removal is a direct calculation | Not learning during use |
| Deleting from language-model memory | Ramesh [10] | Subtract, transport or replay to delete from in-context memory | Weights never change; transport fails when memory spans many layers |
| Tracing answers to data | TracIn [25] | Approximate influence of training examples over checkpoints | Approximate; STARK's RQ2 baseline |
| Retrieval | RAG [23] | Exact, easy deletion from an index; answers can cite passages | The model learns nothing; questions that need several circulars are hard to fetch in full. STARK's main baseline |
| Attacks | Test-time adaptation poisoning [11, 12, 13] | Poisoning of learning during use, shown for image models | Not language models |
| | Indirect prompt injection [26] | The threat model: instructions hidden in content the agent reads | Only while the text is in context |
| | Agent memory security survey [15] | Attacks on agent memory | Leaves memory held in weights out of scope |
| Compliance question answering | Abualhaija et al. [27] | How to evaluate question answering over regulations | No learning |
| Law | Veale et al. [28] | A trained model can itself be personal data, so erasure must reach the weights | No method |

Our literature search (October 2026) found **no study of injected instructions persisting through test-time learning in language models.**

### The gap STARK fills

| Borrowed, with credit | New in STARK |
| --- | --- |
| Learning in `down_proj` during use [1] | Measurement of how injected instructions persist under test-time learning in language models (RQ1) |
| The teacher-student update rule [2] | Exact tracing of answers to the inputs that taught them, for test-time learning (RQ2) |
| Logging updates [6]; checkpoint and replay for exact deletion [4, 7, 10] | Exact removal for test-time learning, including the leak through the teacher's view, with recomputation limited to one teacher window (RQ3) |
| Keeping the learnable part linear on a frozen model [8, 9] | A design (one learning layer, recorded teacher answers) under which the one-step removal shortcut is exact, where it fails for memory spread over many layers [10] |
| Carrying an update forward through later steps [10] | A benchmark built from real RBI circular withdrawals, planted fake notices and made-up customer records |

## Rough shape of the solution

A frozen open model of about 1B parameters, with one matrix that learns while it reads. Every update is written to a **lesson ledger**. Because the learning matrix sits on top of frozen layers and the teacher's answers are recorded as fixed numbers, the ledger can be replayed exactly. Replay makes two things possible: tracing any answer to the entries and sources behind it, and removing any source so that the result equals a run that never read it.

## Evidence we already have

- **The worked example.** The fake-notice example in the documentation (section 7) was computed by hand and checked in code. It is now `tests/test_toy.py`.
  - Deleting only the notice's own entry still leaves Q2 at −0.15, because of the leak.
  - Removing the notice as a source gives Q1 +0.59 and Q2 +0.45, identical to a run that never saw it.
- **One-step removal equals replay.** On random matrices, the one-step removal shortcut and a replay without the entry agree to within rounding. This is now `tests/test_properties.py`.
- **Feasibility arithmetic for a 1B model.**
  - Ledger: about 1.5 MiB per entry of 64 tokens.
  - Replay: about 4.3 billion operations per entry.
  - Tracing: about 2 million operations per entry for each answer position.
  - Nothing backpropagates through the network to learn.

## What we still don't know

These are the project's main risks. The first three phases' gates are designed to answer them.

1. **Does one learning layer learn enough to beat context-only memory?** Tested at Gate 1. If not, add layers and report the measured approximation.
2. **Does learning beat RAG on RBI compliance questions?** If not, there is less reason to update weights. The ledger still delivers tracing and removal, and could move onto a LoRA adapter if needed. This is the top risk reported to the guide on 14 October.
3. **Is exact removal fast enough, and exact on real GPUs?** Replay grows with session length, and GPU results can differ slightly between runs.
   - Fallbacks: checkpoints keep replay short, and the code runs in deterministic mode.
   - If it is still too slow, subtract the logged update instead, and measure what is left against exact removal.
4. **How strongly and for how long do injected instructions persist?** This is RQ1, and the answer is a finding, not a risk.
5. **How much does hiding a source differ from deleting it?** Hiding keeps later tokens at their original positions. The effect is expected to be small, and it will be measured.

## References

[1] G. Feng, S. Luo, K. Hua, G. Zhang, D. He, W. Huang, T. Cai. In-Place Test-Time Training. ICLR 2026. arXiv:2604.06169.

[2] Z. Wang, X. Dang, R.-J. Zhu, Z. Wen, H. Fu, W. Chai, J. D. Lee. Learning What to Remember: Test-Time Training via Context Distillation. 2026. arXiv:2608.01672.

[3] A. Behrouz, P. Zhong, V. Mirrokni. Titans: Learning to Memorize at Test Time. 2025. arXiv:2501.00663.

[4] L. Bourtoule, V. Chandrasekaran, C. A. Choquette-Choo, H. Jia, A. Travers, B. Zhang, D. Lie, N. Papernot. Machine Unlearning. IEEE Symposium on Security and Privacy, 2021. arXiv:1912.03817.

[5] J. Xu, Z. Wu, C. Wang, X. Jia. Machine Unlearning: Solutions and Challenges. 2024. arXiv:2308.07061.

[6] L. Graves, V. Nagisetty, V. Ganesh. Amnesiac Machine Learning. AAAI 2021. arXiv:2010.10981.

[7] Unlearning at Scale: State-Exact Trace-Preserving Deletion in Billion-Parameter Language Models. arXiv:2508.12220.

[8] Y. Quan, W. Wu, G. Montana. Exact Federated Continual Unlearning for Ridge Heads on Frozen Foundation Models. 2026. arXiv:2603.12977.

[9] A. Golatkar, A. Achille, A. Ravichandran, M. Polito, S. Soatto. Mixed-Privacy Forgetting in Deep Networks. CVPR 2021.

[10] V. Ramesh. Subtract, Transport, or Replay? Auditable Deletion from Language-Model Memory. 2026. arXiv:2607.27539.

[11] T. Wu et al. Uncovering Adversarial Risks of Test-Time Adaptation. ICML 2023.

[12] On the Adversarial Risk of Test Time Adaptation: An Investigation into Realistic Test-Time Data Poisoning. ICLR 2025. arXiv:2410.04682.

[13] S. Pi, X. Wang, J. Pi. Research on the Robustness of the Open-World Test-Time Training Model. Frontiers in Artificial Intelligence, 2025.

[14] F. Barez. When AI Systems Learn During Deployment, Our Safety Evaluations Break. Oxford Martin AI Governance Initiative, January 2026.

[15] A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle. 2026. arXiv:2604.16548.

[16] F. Chollet. On the Measure of Intelligence. 2019. arXiv:1911.01547.

[17] Regulation (EU) 2016/679 (General Data Protection Regulation), Article 17; Digital Personal Data Protection Act, 2023 (India); Regulation (EU) 2024/1689 (Artificial Intelligence Act).

[18] Reserve Bank of India warning on fraud using fake RBI letterheads and email addresses. News On AIR, 29 August 2024.

[19] Reserve Bank of India. Circular DOR.RRC.REC.302/33-01-010/2025-26, 28 November 2025: withdrawal of 9,445 circulars on consolidation into 244 Master Directions.

[20] Reserve Bank of India. Annual Report 2024-25: 353 penalties in 2024-25. Reported by Business Standard, 1 June 2025.

[21] Reserve Bank of India. Report of the Committee on the Framework for Responsible and Ethical Enablement of Artificial Intelligence (FREE-AI), 13 August 2025.

[22] Ministry of Electronics and Information Technology. Digital Personal Data Protection Rules, 2025, notified 13 November 2025.

[23] P. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020. arXiv:2005.11401.

[24] K. Meng, D. Bau, A. Andonian, Y. Belinkov. Locating and Editing Factual Associations in GPT (ROME). NeurIPS 2022. arXiv:2202.05262.

[25] G. Pruthi, F. Liu, M. Sundararajan, S. Kale. Estimating Training Data Influence by Tracing Gradient Descent (TracIn). NeurIPS 2020. arXiv:2002.08484.

[26] K. Greshake et al. Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. ACM AISec 2023. arXiv:2302.12173.

[27] S. Abualhaija et al. Automated Question Answering for Improved Understanding of Compliance Requirements. IEEE RE 2022.

[28] M. Veale, R. Binns, L. Edwards. Algorithms that Remember: Model Inversion Attacks and Data Protection Law. Phil. Trans. R. Soc. A, 2018. arXiv:1807.04644.

Also kept as an RBI test set: IndiaFinBench (Pall, arXiv 2026).
