# Research 612: Wave-Based Semantic Memory / ResonanceDB — PASS

> **Source:** [Wave-Based Semantic Memory with Resonance-Based Retrieval: A Phase-Aware Alternative to Vector Embedding Stores](https://arxiv.org/abs/2509.09691) — Aleksandr Listopad, arXiv:2509.09691v1, 21 Aug 2025, cs.IR
> **Date:** 2026-10-08
> **Status:** PASS (no adoption; prior-art + internal-superset coverage; filed as a note at owner request — the PASS-Redirect one-liners are the committed record, this note is the full verdict transcript)
> **Verdict record:** Claude ping-pong session `8e75c9e8` — AGREE round 1 (three refinements applied), closed round 2. Redirects committed in `cb7f485d1`.
> **Classification:** Public

---

## TL;DR

Complex waveforms ψ(x)=A(x)e^iϕ(x) retrieved by a "resonance score" claimed to beat cosine on
negation/phase-shift/intensity queries. **The score is a known-parts assembly**: verified
against the paper's §2 score definition,

S = ½·(Σ|ψ1+ψ2|²)/(Σ(|ψ1|²+|ψ2|²))·R, with R = 2√(E1E2)/(E1+E2),

reduces to **S = ½β(1+β·cos_h)** where cos_h = Re⟨ψ1,ψ2⟩/√(E1E2) is **Plate's FHRR published
VSA similarity** (1994/2003), β·cos_h = 2Re⟨ψ1,ψ2⟩/(E1+E2) is the **soft Dice coefficient**
(Sørensen/Dice; soft form V-Net arXiv:1606.04797), and β is the **Michelson interference
fringe visibility**. The interleaving (A·cosϕ, A·sinϕ) makes the score plain real cosine on a
2L vector rescaled by an energy weight — order-equivalent to cosine for equal-norm
candidates; β re-weights only when norms differ. Empirical claims are circular (phases
procedurally constructed to encode NEG/SHIFT operators, then retrieved as phases; the paper's
own §Limitations: "no learnable phase modulator"). 1 tangential citation, no reproduction.

---

## 1. Why PASS (three independent kills)

1. **External prior art** (subagent sweep, ~14 searches + Semantic Scholar): the score is an
   assembly of three published components (FHRR similarity — confirmed as published VSA
   practice; soft Dice; fringe visibility). RoPE (arXiv:2104.09864) is rotation-based
   similarity at scale. No single kill needed.
2. **Internal superset** (the load-bearing one): katgpt-core `geometric_product`
   (Research 299, Super-GOAT, DEFAULT-ON, G1 +17.6pp non-redundancy) computes uv = u·v + u∧v —
   the paper consumes only the scalar (real) half of the complex product z1·z̄2; the imaginary
   half Im Σ z1z̄2 is a contraction of 299's bivector. ℂ is the minimal Clifford algebra: the
   wave score is a strict subset of the shipped primitive under a different normalization.
3. **Circular evaluation + no real-text phase constructor**: on the sign-phase compat path
   (ϕ ∈ {0,π}) the score is order-equivalent to cosine (equal-norm); the P@1=1.0 wins exist
   only where the experimenters constructed the phases. The one genuinely missing piece
   everywhere (a real-text phase/stance constructor) is exactly what the paper does not provide.

## 2. Internal surface audit (what was checked, post-AGREE addenda)

- **riir-rag `ModellessEmbedder`** (authoritative copy `riir-neuron-db/crates/riir-rag` —
  re-verified 2026-10-08): the DFT dim computes re/im accumulators then takes MAGNITUDE only
  (`|X[1]|/Σhist`); `atan2`/phase occurs ZERO times in the file. The paper's "phase-blind
  retrieval" critique is literally instantiated in our substrate — and is CORRECT behavior
  there: byte-histogram phase is noise, and LATENT_DIM=8 is a pinned `ShardIndex` contract.
  (The stale pre-carve copy in riir-ai/crates/riir-rag that this session first read is
  dead-tree residue — riir-ai Issue 1040, filed 2026-10-08.)
- **riir-refine RuleIndex** (the paper's closest consumer surface): BM25 + latent fusion;
  the documented limitation (latent term "close to a hash" on the clippy corpus, T3.5 sweep)
  is BLAKE3-avalanche zero-locality — a phase channel on a hash is still a hash — mitigated
  by the shipped Bonsai dense rerank. `score_norm.rs` already ships energy/extremes-aware
  fusion shaping (β knob, degenerate_credit, RRF-vs-margin tests).
- **riir-rethink** (the trained-encoder product): NO retrieval/embedding store exists
  (grep `knn|nearest|retriev|embedding_store|cosine|dot_product` over `src/` — zero hits);
  decisions come from heads over the forward pass, so the paper has no surface there. Ironic
  addendum: the laya encoder's attention ALREADY runs phase-rotation similarity — RoPE
  (`rope_theta_full`/`rope_theta_slide`, `apply_rope` rotate-half on q/k in every forward,
  CPU/Metal/CubeCL/CUDA) — the paper's mechanism class in its standard, G5-gated form, with
  the phase supplied by the one constructor that works: a deterministic rule (position).
- **reflex `CorpusDistanceGate`**: threshold-calibrated from measured geometry, G1-gated;
  β-damping would be a config tweak with no documented failure mode — not actionable.

## 3. The residual gap + the one fusion candidate (filed, TBD)

**Re-open trigger:** a real-text phase/stance constructor (procedural or learned) that
measurably separates negation/stance in actual LLM embeddings. If one appears, re-evaluate —
our retrieval surfaces (RuleIndex, ShardIndex, a future rethink embedding memory) would then
have the missing ingredient.

**Filed candidate** — riir-refine Issue 157 (2026-10-08): span-conditional contrastive
negatives in RuleIndex — the v4 `rejected` labels are span-conditional contrastive negatives
already computed client-side, but demotion today is rule-GLOBAL (`EvidenceTier::Withdrawn`).
A per-span negative memory (dot-gated sigmoid down-weight — the paper's "anti-phase" in
real-vector form) is the one fusion idea this analysis surfaced. Novelty TBD, owner triage;
not paper-derived (standard contrastive retrieval) — the paper's framing only exposed the gap.

## 4. PASS-Redirects (committed `cb7f485d1`, 2026-10-08)

- Research 299 (Clifford Geometric Product) — primary: the strict-superset argument + the
  reduction formula (R included so future readers can verify without the PDF).
- Research 305 (Phase-Modulated Cross-Domain Coupling) — the phase-family pointer (the
  rotation-gate cousin; RoPE note above reinforces it).
- Research 169 was considered and DROPPED (verdict round 1): its body is LinOSS/PDE
  dynamics, not a similarity substrate — a redirect there would be grep noise.
