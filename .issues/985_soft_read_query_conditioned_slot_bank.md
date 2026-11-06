# Issue 985 — Sigmoid-gated query-conditioned fixed-K soft read with deterministic slot banks (katgpt-core primitive)

**Status:** CLOSED NEGATIVE — landed + struck 2026-11-06, commit `<this>`; delegated verdict AGREE (claude_code session `a842c972-dfca-4189-9108-244817f2d5b3`). T1–T7 all executed; the preregistered bar FAILED at the amended cell; the grade is struck and the feature stays opt-in for replication. The two mechanism hypotheses are filed separately as Issue 986.
**Lease closed:** zed-m3-agent (T1–T7), 1793929810, expires 2026-11-08T01:50:10Z — completed within lease.

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

## Amendment 3 (2026-11-06, measured at the first GOAT run — recorded BEFORE the rerun)

The first cell run FAILED its own fixture discipline (hard16(c=2) = 1.0000 — no collapse at c=2) and exposed a posture defect. Three amendments, measured reasons, no tune-until-win:

1. **Distractors per chain 8 → 18** — measured: with 12 own-chain rows (4 hops + 8 distractors), nothing else in the store ranks above hop-2's cosine 0.507, so f_2 leaks INTO hard's top-16 and the collapse cannot exist. 19+ same-topic rows above hop-2 saturate the 16 slots first (f_1 + 15 distractors).
2. **Gate bias b = −2 (the b_k dial, armed)** — measured: at b = 0, σ(0) = 0.5 hands every zero-cosine row HALF gate, so the gated average collapses toward the global store mean (soft best-answer cosine 0.54/0.27/0.21/0.29 — flat, non-selective). b_k exists in the preregistered formula as the slot's selectivity dial; −2 puts the 0.5-gate point at cosine 0.2.
3. **Extraction semantic: per-slot top-gate row, best over slots (soft16 vs hard16, same 16-candidate budget)** — measured/structural: the previous soft score (best slot AVERAGE's direction vs hard's best-of-16 EXACT row) is asymmetric by construction — an average of a neighborhood cannot reach an exact row's cosine of 1.0 even when the read is perfect. The modelless "consumer picks the useful token" semantic is the top-gate row per slot (argmax_i g_ik = argmax_i ⟨q_k, z_i⟩ — gate-scale-free by sigmoid monotonicity); 16 slots × top-1 = the same candidate budget as hard's top-16, so the ONLY difference is conditioning. The average-direction read stays PRINTED as the aggregate reference, never gated.

Store-size arm levels clamp to the store (N=1 tests level 1 only; the chain object always carries 4 hops, the STORE carries what fits). Margin/tol on the G3 quality-direction inequalities widened to 0.05 (best-of-16-gated-row cosine of a noise-store read has E ≈ 0 but σ ≈ 0.1 per replicate — 0.02 would flake, 0.05 is the honest equality band at this metric's scale).

If the amended cell still loses to hard16, the negative is recorded and the grade struck — no second amendment pass.

## Prior odds (disclosed, not hidden)

1. The paper never ran an untrained FIXED read — the floor's quality is a bet, not a fact.
2. Reflex corpus-cap accuracy levers previously measured NEGATIVE (Issue 013) — corpus-summary features have a bad track record on this stack.
3. Research 606 HyperThink (query-conditioned amortization) is a recorded negative on this workspace — different mechanism, but a caution against assuming query-conditioned latent features transfer automatically.
If the amended cell still loses to hard16, the negative is recorded and the grade struck — no second amendment pass.

## The verdict (measured, after amendment 3)

**G1 composition (N=512, best-answer cosine per level; soft16 = per-slot top-gate row, same 16-candidate budget as hard16):**

| c | soft16 | soft-avg | soft-orth | hard16 | hard16-mean |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.8498 | 1.0000 | 1.0000 | 0.9619 |
| 2 | 0.3770 | 0.5022 | 0.3770 | 0.3983 | 0.4035 |
| 3 | 0.1941 | 0.2788 | 0.1941 | 0.2271 | 0.1647 |
| 4 | 0.2415 | 0.3467 | 0.2415 | 0.3230 | 0.2409 |

The fixture discipline HOLDS (hard16 collapses 1.0 → 0.398 → 0.227 → 0.323 — the fixture is valid), and soft16 loses to hard16 at every non-trivial level. The preregistered bar fails.

**G3 Issue 984 arms (verdict-narrowed instrument — both levels, hard16 under every arm, unique-candidate count, head-alignment spread):**

| level | store | soft16 | hard16 | uniq-cand | head-align cos(q_k,q) |
|---|---|---|---|---|---|
| c=2 | real | 0.3770 | 0.3983 | 1.0 | +0.9970 |
| c=2 | noise | 0.1967 | 0.2875 | 1.2 | +0.9967 |
| c=2 | shuffled | 0.1154 | 0.2822 | 1.0 | +0.9978 |
| c=4 (gated) | real | 0.2415 | 0.3230 | 1.0 | +0.9970 |
| c=4 (gated) | noise | 0.2878 | 0.3037 | 1.2 | +0.9967 |
| c=4 (gated) | shuffled | 0.1028 | 0.2212 | 1.0 | +0.9978 |

Deepest level: I1 FAIL (noise 0.2878 ≥ real 0.2415 — the noise store BEATS the real store), I2 FAIL (noise > empty + tol), I3 OK. c=2 discloses I1 OK (real 0.377 > noise 0.197) — the read does carry SOME signal at levels with real-store support; the deep level barely discriminates for ANY method (hard16-real 0.323 vs hard16-noise 0.304).

**G4 second signal:** W1 AURC soft 0.2535 vs score-only 0.2103 — the query-conditioned reconstruction confidence makes ranking WORSE than ignoring it; W2 correctly shows no signal. No consumer case.

**Mechanics (all pass — the negative is about read quality, not broken code):** replay bit-identical; zero-alloc steady state; zero-gate ε→0; boundedness at N ∈ {1,16,512}; coverage entropy 0.984.

## The findings (verdict-narrowed wording)

1. **Conditioning strength is the binding failure (MEASURED):** head-alignment cos(q_k, q) = +0.997 on EVERY store — the β-residual update moves each head by ~0.3%, so all 16 heads are effectively the raw query, and unique-candidate count = **1.0** (16 slots, ONE distinct pick vs 1.2 on noise). A 1-candidate read loses to 16-candidate hard top-16 by construction. E[max cos of 16 random draws] ≈ 0.22 at D=64; the noise arms' 0.29-0.30 is materially above that — head-ALIGNED selection, not pure random draw.
2. **Placement (HYPOTHESIS, not tested here):** FPS anchors maximize min-distance → they spread to the store's outskirts; the interior hop rows the query needs get no anchors. Recorded as a hypothesis; Issue 986 tests it (density-aware deterministic placement) against a bar of hard top-16 under the 984 arms.
3. The deep level (c=4) barely discriminates for any method — the honest multi-hop signal lives at c=2/3, where soft16 still loses (0.377 vs 0.398, 0.194 vs 0.227).
4. The paper's premise SURVIVES this negative: Memorilla's slots are LEARNED, and what learning buys here is exactly what the modelless floor lacks (effective conditioning magnitude + placement). The trained lane (riir-train Plan 488 R1) is NOT indicted — its preregistered bar compares to hard top-16, which this run confirms as the standing champion on the fixture class.

## Demote execution (this commit)

- Feature `soft_read` stays opt-in, labeled MEASURED NEGATIVE (Cargo comment, module doc, bench header).
- `bench_985_soft_read_goat.rs` = REPORT-ONLY replication instrument (gates print, exit 0); ONE live assert: fixture validity (hard16 collapse must exist, or the negative is void and the cell re-opens). Manual bench — required-features armed, no gate row needed.
- `soft_read::distance_soft` (the CorpusDistanceGate second signal) = replication code, NOT a consumer landing. The max-sim path and bench_845 are untouched (G1–G6 re-verified green during the shared-worlds refactor).
- Mechanics tests stay as the named row `soft_read_gates` (they pin the determinism/alloc/ε semantics of the kept-for-replication primitive).
- The `soft_read` feature is NOT in any default set and gains no consumer.

## Tasks

- [x] T1: `slot_bank` module — FPS + Hadamard constructors, BLAKE3(store)-keyed, K>N orthonormal fill, coverage-entropy gate (0.984 measured)
- [x] T2: `soft_read` primitive — conditioned heads, sigmoid gates, NORMALIZED read ε=1e-6, zero-alloc scratch API, const-generic K
- [x] T3: LOCAL-ONLY determinism comment at the code site (Issue 940 law) + replay bit-identity test
- [x] T4: composition-synthetic fixture (1→4 collapse shape; fixture discipline verified: hard16 collapses) + hard top-16 baseline + empty floor
- [x] T5: store-size arm N ∈ {1,16,512} + zero-gate test (N=16 noted: top-16 of a 16-row store is every row — bounds soft from below, cannot distinguish mechanisms)
- [x] T6: CorpusDistanceGate second-signal consumer (composed, gate file untouched) + W1/W2 re-run — the second signal FAILS W1 (AURC worse than score-only); recorded
- [x] T7: Issue 984 arms attached + verdict table + GOAT cell run `--release` + **DEMOTE-ON-FAIL executed**: verdict NEGATIVE, recorded here and in Research 656
