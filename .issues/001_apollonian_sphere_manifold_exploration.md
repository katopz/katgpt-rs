# Issue 001: Apollonian Sphere Manifold Geometry — Exploration

**Date:** 2026-06-23
**Status:** Closing — strongest use case (FUNCATTN basis selection, #4) evaluated and rejected on prior evidence (2026-06-26). MMORPG use cases (#1-#3) rejected on domain-shape mismatch (2026-06-26). Close on schedule 2026-07-23 unless new evidence emerges.
**Origin:** Gemini "Functional Attention + Relational Functor" reframing (2026-06-23)
**Related Research:** katgpt-rs/.research/257 (FUNCATTN), katgpt-rs/.research/219 (TNO/DEC), katgpt-rs/.research/291 (cross-resolution transport), katgpt-rs/.research/100 (EGA — fixed<learned precedent)

## Context

The Gemini reframing of our latent-to-latent pipeline proposed "nested Apollonian
topologies" as the manifold geometry underlying our latent space. Apollonian
sphere packings (Graham–Lagarias–Mallows–Wilks 2003) have real mathematical
properties that flat `R^d` lacks:

- **Hierarchical metric structure** — natural parent–child relationships between
  packed spheres at multiple scales.
- **Self-similarity** — same packing structure recurs at every scale.
- **Multi-resolution decomposition** — coarse packings are limit approximations
  of fine packings by construction.
- **Known harmonic decompositions** — Apollonian group structures connect to
  spherical harmonic analysis (relevant to FUNCATTN basis selection).

Grep across all 5 repos confirms **zero hits** for "Apollonian" — genuinely
unexplored in our corpus. Not present in `katgpt-rs/.research/`,
`riir-ai/.research/`, `riir-chain/.research/`, `riir-neuron-db/.research/`, or
shipped code.

## The Question

**What concrete game-AI or shard-retrieval use case does Apollonian geometry
enable that our current flat `R^d` + dot-product + sigmoid projection does not?**

Candidate use cases (each needs validation before this can become a plan):

1. **Hierarchical shard retrieval** — Apollonian packing gives natural
   parent–child metric relationships. Could `ShardIndex` use this for
   multi-resolution zone→shard lookup? *Baseline to beat:* lock-free
   `papaya::HashMap` O(1) lookup at current zone count.
2. **Cross-resolution personality transfer** — if shards live on an Apollonian
   manifold, small-dim shards are "coarse approximations" of large-dim shards
   by construction. *Related:* Research 291 (cross-resolution spectral transport).
3. **NPC social hierarchy** — Apollonian packings have a natural
   "center vs periphery" structure. Could this model faction hierarchies or
   attention allocation? *Baseline to beat:* current zone-density gating
   (`latent_functor/zone_gating.rs`).
4. **Spectral basis selection for FUNCATTN** — Apollonian packings have known
   harmonic decompositions. Could this give a better basis than spherical
   harmonics for FUNCATTN? *Baseline to beat:* sigmoid-normalized learned basis
   at k=4..16 (Research 257 §5.5).

## Why This Is an Issue, Not a Plan

We cannot run the novelty gate (Q1–Q4) honestly without a concrete use case.
"Nice geometry" alone fails Q3 (product selling point) and likely Q2 (new class
of behavior — it's not obviously a new capability vs an optimization). Once a
use case is proposed where Apollonian geometry beats flat `R^d` on a measurable
metric, this promotes to a plan with a real GOAT gate.

## Success Criteria (to close this issue)

- [x] Propose a concrete use case with a measurable metric — **DONE 2026-06-26**:
      use case #4 (FUNCATTN basis selection) evaluated with concrete task W
      (multi-scale synthetic transport, d=64, n=20, k∈{4,8,16}), metric
      (reconstruction cos ≥ 0.85), baselines (random-orthogonal, PCA, learned).
- [x] Sketch the minimal prototype — **DONE 2026-06-26**: replace `W_basis` with
      pre-computed Apollonian harmonics, benchmark vs random-orthogonal on the
      multi-scale transport task. See §"Evaluation" below.
- [x] Identify a kill condition — **DONE 2026-06-26**: hard kill = Apollonian cos <
      random-orthogonal at any k; soft kill = Apollonian cos < learned
      data-adaptive. **Kill triggered on prior evidence** (see §"Evaluation").

If no concrete use case is proposed within 30 days (by 2026-07-23), close as
"shelved — no concrete payoff identified". Do not let this linger as
perpetually-open speculative math.

## Evaluation (2026-06-26)

Both use case families (MMORPG #1-#3, FUNCATTN #4) were evaluated and rejected.

### MMORPG use cases (#1-#3) — rejected on domain-shape mismatch

Apollonian geometry answers "have metric, want hierarchy". MMORPG domains are
the inverse: factions/zones/social are explicit trees/graphs (structure known,
metric wanted); positions must stay raw flat-R² by anti-cheat rule; emotions use
flat dot-product+sigmoid by rule. Every candidate either loses to an existing
flat baseline (`papaya::HashMap` O(1), `latent_functor/zone_gating.rs`) or isn't
gameplay-native. Forcing it would fail the GOAT G1 (correctness) and Q3 (selling
point) by construction.

### FUNCATTN basis selection (#4) — rejected on three independent precedents

Concrete design proposed: replace `W_basis ∈ R^{d×k}` (currently caller-supplied
random-orthogonal) with pre-computed Apollonian harmonic columns. Benchmark on
multi-scale synthetic transport (d=64, n=20, k∈{4,8,16} — the open sweep from
Research 257 §5 item 5). Metric: reconstruction cos ≥ 0.85 vs random-orthogonal,
PCA eigenbasis (T5.1 known-fail), and learned data-adaptive (EGA winner).

**The design does not hold up. Three codebase-grounded precedents kill it:**

1. **T5.1 null result (Plan 286 L145-146)** — SpectralQuant PCA eigenbasis
   pre-rotation was 17-25% WORSE than vanilla. Root cause (verbatim): "the
   adaptive basis's row-normalization is invariant to basis direction — rotating
   the rows doesn't concentrate information, it just rotates the score frame."
   Apollonian harmonics are an orthogonal-ish fixed rotation → same failure mode.
2. **EGA negative result (Research 100 L33-34, L39-49)** — fixed Morlet +0.001,
   fixed db2 +0.005, fixed db4 -0.001, vs learned EGA-1 +0.103. Verbatim: "Fixed
   wavelets (Morlet, Daubechies) are near-baseline. Only the learned
   data-adaptive projection works." Apollonian harmonics share the defining
   property of the losers (fixed, data-independent, multi-scale) — no mathematical
   reason to expect them to escape.
3. **FUNCATTN G6 already failed (Plan 286 T4.4)** — FUNCATTN 0.969 < SDPA 1.000
   on masked-token LM prediction. Rescuing a failed primitive with a fancier
   fixed basis is a non-GOAT; even a match would only move FUNCATTN from "failed
   G6" to "still failed G6, marginally less".

The strongest *a priori* argument for Apollonian is its hierarchical multi-scale
structure, but: (a) cross-resolution transport (Plan 310, DEFAULT-ON) already
handles multi-scale via asymmetric *learned* bases + BLAKE3 commitment, better;
(b) FUNCATTN's k×k transport operator C is flat and doesn't exploit hierarchy —
benefiting from Apollonian's structure would require a block-structured C, which
is a *different primitive* (hierarchical transport), not basis selection;
(c) latent space has no demonstrated sphere-packing structure, so Apollonian is
a geometry-as-bias applied to a space that may not have that geometry.

**Verdict**: no measurable benefit identified. Running the experiment would
almost certainly produce a third documented negative result (after T5.1 and EGA).
Close on schedule 2026-07-23 unless new evidence emerges.

### Paths that ARE worth pursuing (not Apollonian)

- **Learned data-adaptive basis selection** (the EGA winner) — a runtime basis
  quality metric + freeze/thaw swap of learned `w_basis` per domain. This is the
  direction the evidence points, and it's modelless (freeze/thaw path #1).
- **k-sweep for the NPC regime** (Research 257 §5 item 5) — run the open
  d=64/n=20/k∈{4,8,16} sweep with the EXISTING sigmoid basis. No new geometry
  needed; just fill the documented gap.
- **Hierarchical transport operator** (block-structured C) — if multi-scale
  structure in the operator itself is the goal, that's a new primitive, not
  Apollonian basis selection. File separately if pursued.

## Related Work (external, TBD)

- Graham, Lagarias, Mallows, Wilks, Yan — *Apollonian Circle Packings: Number
  Theory* (2003). J. Number Theory 100:1–45.
- Spherical harmonics on Apollonian packings — analysis literature (needs
  targeted arxiv search before promoting to plan).
- Hyperbolic embeddings (Poincaré, Nickel–Kiela) — different non-Euclidean
  geometry but same "geometry-as-inductive-bias" thesis; relevant prior art for
  the "is non-flat geometry worth it?" question.

## Cross-Refs

- `katgpt-rs/.research/257_Functional_Attention_Spectral_Transport_Operator.md`
  — FUNCATTN basis selection (use case 4).
- `katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md`
  — DEC operators on cell complexes (different topology, same
  "geometry-as-routing" idea — shows we're already open to non-flat geometry).
- `katgpt-rs/.research/280_Resolution_Tiered_Deterministic_Commitment.md`
  — resolution tiering (related to use case 2).
- `katgpt-rs/.research/291_cross_resolution_spectral_transport_open_primitive.md`
  — F3 fusion target (Apollonian as the natural multi-resolution basis).
- `riir-neuron-db/src/index.rs` — `ShardIndex` baseline for use case 1.

## TL;DR

Apollonian sphere packings proposed as latent manifold geometry. Zero hits in
our 5-repo corpus — genuinely unexplored. Cannot run the novelty gate without a
concrete use case. File as exploration; promote to plan only when someone
proposes a measurable win over flat `R^d`. Close in 30 days if no use case
emerges.
