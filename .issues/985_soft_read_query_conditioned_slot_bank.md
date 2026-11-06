# Issue 985 — Sigmoid-gated query-conditioned fixed-K soft read with deterministic slot banks (katgpt-core primitive)

**Status:** OPEN — fusion idea with preregistered proof path, filed from Research 656 (arXiv:2610.21375 Memorilla distillation, delegated-owner AGREE 2026-11-06). NOT a Super-GOAT claim — the floor's quality is unproven (see Prior odds); this issue exists to make the bet cheap and the losing loud.
**Claimed:** zed-m3-agent (T1–T7), 1793929810, expires 2026-11-08T01:50:10Z

## Mechanism (modelless floor form of Memorilla's read)

K read heads over a persistent store, no learned weights:

- **Slot bank (deterministic):** K=16 store-derived anchors — farthest-point/k-means++ anchors over the store (ON the data manifold), or BLAKE3-seeded orthonormal Hadamard directions (geometry-partitioning views). Computed once per store epoch, keyed `BLAKE3(store)`, cached; incremental-append re-derivation pinned. (Replaces Memorilla's Xavier-learned slots; the paper's residual design means slots need to be mutually DISTINCT, not pretrained.)
- **Conditioned head:** `q_k = normalize(z_q + β_k · ⟨z_q, d_k⟩ · d_k)` — each head reads the store through its own deterministic view of the query (slot distinctness lives in the residual path, per the paper's Eq. 3 note that a single-vector query gives all slots a shared update).
- **Gate (sigmoid, never softmax — house law):** `g_ik = σ(⟨q_k, z_i⟩/τ + b_k)`.
- **Read (NORMALIZED — this is load-bearing, see Normalisation):** `M_k = Σ_i g_ik·z_i / (Σ_i g_ik + ε)`.

## Normalisation (amendment, reviewer round 1)

A sigmoid-gated SUM grows with store size; Memorilla's flat-cost + quality-from-1-to-512-docs property depends on normalisation. The read is therefore a **gated weighted average**: bounded by the longest stored vector by construction. **ε = 1e-6, fixed relative to the gate scale** (gates live in (0,1), so Σg ranges up to N; 1e-6 is ≪ any non-degenerate gate mass yet keeps the denominator strictly positive without meaningful shrinkage — the division scales M_k down by a factor ≈ Σg/(Σg+ε), i.e. ~1 part in 10⁶ at Σg=1). When the gate total is much smaller than ε the read shrinks toward the zero vector — the CORRECT behaviour: "nothing relevant" must look like an empty memory. G1 asserts exactly that (zero-gate test below).

## First consumer (amendment, reviewer round 1)

**`CorpusDistanceGate`** (`crates/katgpt-core/src/distance_abstain.rs`): add the query-conditioned aggregate as a SECOND signal beside max-similarity, re-run through the Bench 845 W1/W2 worlds (`crates/katgpt-core/benches/bench_845_distance_abstain_goat.rs` — full path mandatory in any gate row; katgpt-attn's unrelated `bench_845_channel_aware_dot_dispatch.rs` is the reason). A loss shows up on day one; bench_845 stayed opt-in partly for lack of a live consumer — this is the cheapest one. Later consumers (listed, not promised): refine `RuleIndex` span-read rerank; NPC think-brain ring read (riir-ai, rides riir-train Plan 488 R1's floor arm).

## Determinism law (Issue 940; amendment, reviewer round 2)

The aggregate is read LOCAL-ONLY — never committed, exported, hashed, or synced. **Record that decision at the code site** as a comment; if a future consumer ever syncs it, the fold order must be pinned AT THAT POINT (sequential fold, fixed tie rule), never inherited by accident.

## Pre-registered gate

- **Bar: ≥ hard top-16 AND > empty-memory floor.** Hard cosine top-16 (query-aware, discrete) is the adversarial baseline — the paper's nearest untrained ablation captured +4.9 of +15.1, so the floor must beat DISCRETE selection, not the trained module. The empty-memory floor (no primitive at all) is the second bar: the paper's random-init result (0.153 vs 0.186 no-memory) proves untrained latents can lose to NOTHING, and that must fail loudly here too.
- **Composition-synthetic fixture:** slices built to reproduce the paper's Fig 4(c) shape — top-k accuracy collapsing 78.7 → 51.8 as composed-fact count goes 1 → 4 while a query-conditioned read holds. The floor form passes only if it survives the collapse shape better than hard top-k.
- **Store-size arm (amendment):** G1 asserts bounded output + the bar held at **N ∈ {1, 16, 512}**.
- **Zero-gate test (amendment):** a store full of zero gates returns ≈ zero (within tolerance) — the empty-memory floor checked from the read side.
- **Noise arm:** Issue 984's arms are MANDATORY members of this gate (real ≥ noise + margin; noise ≤ empty).
- **Posture:** opt-in feature; feature-off byte-identity; demote-on-fail (grade struck in the same commit if the bar misses).

## Prior odds (disclosed, not hidden)

1. The paper never ran an untrained FIXED read — the floor's quality is a bet, not a fact.
2. Reflex corpus-cap accuracy levers previously measured NEGATIVE (Issue 013) — corpus-summary features have a bad track record on this stack.
3. Research 606 HyperThink (query-conditioned amortization) is a recorded negative on this workspace — different mechanism, but a caution against assuming query-conditioned latent features transfer automatically.

This issue is worth filing ONLY because the gate is cheap (one bench, one fixture, existing worlds) and a loss is decisive information for the whole "query-conditioned aggregate for non-LLM consumers" axis — which the external prior-art sweep found essentially EMPTY (Research 656 §Unclaimed deltas). If the floor loses to hard top-16, the axis closes with a recorded negative instead of a folklore "probably doesn't work".

## Tasks

- [ ] T1: `slot_bank` module in katgpt-core — farthest-point + Hadamard constructors, `BLAKE3(store)`-keyed cache, append re-derivation; slot-coverage entropy gate (even tiling of the store, measured)
- [ ] T2: `soft_read` primitive — conditioned heads, sigmoid gates, NORMALIZED read with ε=1e-6; zero-alloc steady state (scratch-buffer API, const-generic K)
- [ ] T3: local-only determinism comment at the code site (Issue 940 law) + determinism test ((store BLAKE3, z_q) replay bit-identical)
- [ ] T4: composition-synthetic fixture (1→4 composed-fact collapse shape) + hard top-16 baseline + empty-memory floor arm
- [ ] T5: store-size arm N ∈ {1, 16, 512} + zero-gate test
- [ ] T6: CorpusDistanceGate second-signal consumer + Bench 845 W1/W2 re-run (first consumer; keep the existing max-sim path byte-identical when the feature is off)
- [ ] T7: Issue 984 arms attached; verdict table; GOAT cell run `--release`; promote-to-consumer or demote-on-fail per the verdict, recorded in Research 656
