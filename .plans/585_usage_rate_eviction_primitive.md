# Plan 585: Usage-Rate (Mass/Age) KV Eviction Primitive + Generation-Runaway Canary

**Date:** 2026-08-31
**Status:** DONE (2026-09-02) — Phases 1–3 landed; **MIXED GOAT verdict, opt-in** (Bench 697: G1–G4 PASS, G8 regime-bounded — 2–4× raw-H2O recall at cap ≥ 32, one honest miss at the 8%-budget extreme; no consumer = no promotion). Phase 4 stays consumer-pull-gated ([-] below). Record: [`.benchmarks/697_usage_rate_eviction_goat.md`](../.benchmarks/697_usage_rate_eviction_goat.md). **ADDENDUM CLOSED (2026-09-06):** the null-hypothesis arm + protection factorial landed (T3.6–T3.9, `6b840f49`) — signal value CONFIRMED beyond protection (mass_age beats the prompt-pinned random null 5.0×/4.8×/3.8× at cap 32/48/64; collapse non-vacuity confirmed at cap 16), and the standing promotion rule is now `runaway_gate` ∧ `beats_random_prompt_pin` ∧ the protection factorial. G8 verdicts in any future promotion must be read against the controlled null.
**Research:** [katgpt-rs/.research/523_H2O_Norm_Age_Normalized_KV_Eviction.md](../.research/523_H2O_Norm_Age_Normalized_KV_Eviction.md)
**Source paper:** [arXiv:2608.19920](https://arxiv.org/abs/2608.19920) — "Learning how to Forget" (Seeger et al., AWS, 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/kv_eviction/` (new module) + Cargo feature `usage_rate_eviction`
**Track:** PRIMARY (modelless; serving-envelope fit — the score runs at eviction time inside the hot path). The SECONDARY training track is riir-train Plan 367.

---

## Goal

Distill the paper's normalized H2O score into a pure, leaf-clean katgpt-core primitive: per-row usage-rate eviction scoring (`mass / max(1, age)`) over caller-supplied attention-mass increments, with pinned-sink exclusion and per-(b,h) selection by construction — plus the **R/p128 generation-runaway canary**, the output-length diagnostic that gates any lossy KV policy's promotion (the generation-side complement of the Issue 750 lossy-surface rule). GOAT gate: at matched KV budget on a planted age-bias fixture, mass/age retains the hot row raw-H2O evicts, with O(1)/row/step updates and zero steady-state allocation. Not UQ-bearing (no distributions claimed) — the conformal-floor rule does not apply.

House pattern: the primitive consumes caller-supplied observational signal (`suspect_indices(attention_mass, …)` precedent) — katgpt-core stays leaf-clean; mass producers are consumer-side (riir-gpu kernel byproduct → riir-ai Issue 836; HLA recurrent-state probe → Phase 4).

## Phase 1 — Primitive (CORE)

### Tasks

- [x] **T1.1** `crates/katgpt-core/src/kv_eviction/mod.rs` behind `usage_rate_eviction = []` (opt-in): `UsageScoreTable` — per-row `cum_mass: f32` + `admission_tick: u64`, fixed-capacity (caller-owned buffers, zero-alloc; `Vec::with_capacity` once, `clear()`+reuse). *(landed 2026-09-02)*
- [x] **T1.2** `observe(&mut row, mass_increment, tick)` — O(1) accumulate; `score(row, tick) -> f32` = `cum_mass / max(1, tick - admission_tick)` (min-age 1: fresh rows score at their mass, never ÷0; NaN guard: non-finite increment → ignore + debug_assert). *(landed)*
- [x] **T1.3** `select_evict(scores, k, pinned: &[bool]) -> Vec<usize>` — lowest-k scores among unpinned rows, deterministic tie-break by index (ascending) via the `float_order` total-order comparators (`float_order.rs`, NaN-safe — the partial_cmp-unwrap_or intransitivity fix); per-(b,h) by construction (caller slices per head — no cross-head reduction anywhere). *(landed as `select_evict` + zero-alloc `select_evict_into` core — `out` doubles as the workspace; output in eviction-priority order, lowest score first)*
- [x] **T1.4** Property tests: monotone in mass, anti-monotone in age, pinned rows never selected, β=0-pin is a no-op on selection, determinism (bit-identical across runs). *(20 module tests)*
- [x] **T1.5** Reference-parity test vs a naive recomputing implementation on LCG-generated streams (bit-identical scores).

## Phase 2 — Generation-Runaway Canary

### Tasks

- [x] **T2.1** `kv_eviction::canary`: `RunawayStats::from_generations(output_lens: &[usize], target_lens: &[usize], cap: usize)` → `R_median` (output/target ratio), `p_cap` (fraction at cap). Pure fn, zero deps. *(landed; zero-target samples skipped)*
- [x] **T2.2** Encode the promotion rule as a documented fn `runaway_gate(stats, r_max: f32, p_cap_max: f32) -> bool` + doc: **any lossy KV policy (eviction/quantization/compaction) promoted to default MUST pass this gate on a sealed long-context eval** — extends the Issue 750 lossy-surface rule to the generation axis. *(landed; empty-eval FAILS the gate — the vacuity guard)*
- [x] **T2.3** Non-vacuity test: a planted over-eviction fixture must FAIL the gate (fails-before/passes-after — the tile-loop-gate lesson). *(module test + bench-level canary demo: ring@8 R=8.0 FAILS, mass_age@32 R=1.0 PASSES on identical thresholds)*

## Phase 3 — GOAT Bench (falsifiable)

### Tasks

- [x] **T3.1** Planted age-bias fixture: synthetic attention stream with an old-but-cold row (0.001/step × 1000) vs young-but-hot row (0.5/step × 2) at equal raw-H2O cumulative mass — raw-H2O must evict the hot row, mass/age must retain it. The bench is invalid unless this fixture fires. *(PASS: tie arm + strict arm both fire)*
- [x] **T3.2** Micro-GPT long-context recall at matched KV budget (Bench 313 micro-GPT precedent): policies {ring/lastrec β, raw-H2O, mass/age, mass/age+sink-pin, EGA-energy, EGA×usage fusion} — recall/accuracy + `RunawayStats` + eviction-count per policy. *(constructed induction-pair KV, drifted-Zipf workload; 6 policies × 4 caps × 32 seeds — mass_age 50/100/100% vs raw 24/41/51% at caps 32/48/64, miss at cap 16 recorded)*
- [x] **T3.3** Kendall-τ diagnostic: per-head vs batch-summed top-k disagreement over the streams (decides whether per-(b,h) bookkeeping pays; τ ≈ 1 on our workloads ⇒ keep per-head anyway since it is free here, record τ for the kernel-side decision). *(τ = 0.689–0.748 — rankings DISAGREE materially, per-head bookkeeping matters)*
- [x] **T3.4** Gates: G1 determinism (bit-identical double-run); G2 O(1)/row update (criterion, update path < 10ns/row target); G3 default-features no-regression (module fully gated); G4 zero steady-state allocs (TrackingAllocator); **G8 mass/age ≥ raw-H2O recall at matched budget on T3.1+T3.2** — if it loses, keep the negative-result artifact (Bench 697 precedent) and demote. *(G1 PASS; G2 1.22 ns/row PASS; G3 PASS; G4 PASS — after the gate caught a live 1 alloc/step; G8 MIXED: PASS at cap ≥ 32, FAIL at the 8%-budget extreme — regime boundary recorded, primitive NOT demoted since opt-in with no consumer is the standing state)*
- [x] **T3.5** Write `.benchmarks/NNN_usage_rate_eviction_goat.md`; per-stack ledger: slot = KV/eviction; promotion decision (default vs opt-in) per gate outcome + consumer presence. *(Bench 697; decision: OPT-IN — no consumer + regime-bounded G8; promotion re-gate = Issue 836 consumer + real-corpus re-run)*

## Phase 4 — Consumer Probes (pull-gated)

### Tasks

- [-] **T4.1** GPU byproduct kernel (summed attention weights alongside SDPA, cubecl + cudarc twins) → **riir-ai Issue 836** owns the wiring surface; pull-gated on this plan's GOAT pass.
- [-] **T4.2** HLA free-mass probe: linear-attention recurrent state may expose cumulative usage directly (no kernel work) — one probe bench before any kernel investment.
- [-] **T4.3** Replay-log → telemetry → Beta-LCB policy-variant selection (self-adaptive track): substrate exists (katgpt-core `rating`); no serving consumer for policy-variant selection until ≥2 policies run in production — reopen then.
- [-] **T4.4** Content-derived β (`smart_lastrec` regex-prefix variant): paper's own footnote 11 measured the general variant NO better than fixed prefix — fixed β stands; reopen only with a structure-tagged corpus showing fixed-β failure.

## Non-goals

- No runtime wiring in this repo (consumers live in riir-ai — Issue 836).
- No training anywhere (riir-train Plan 367 owns co-adaptation).
- Bonsai-GDN: no KV eviction on recurrent state — out of scope by architecture.

## Addendum (2026-09-04, CLOSED 2026-09-06) — the null-hypothesis arm (Research 531)

arXiv:2609.03430 ("Random Attention", Salesforce) shows prompt-pinned per-head
uniform-random eviction matches the strongest scored evictors on reasoning tasks
at 32–43% higher serving throughput, and that most of the inter-selector gap is
prompt protection, not the score. Bench 697's T3.2 ran six scored/structural
policies with **no random+pin control** — G8's "mass/age ≥ raw-H2O" does not yet
establish signal value beyond protection.

### Tasks

- [x] **T3.6** Add the `random_prompt_pin` arm to the Bench 697 harness: seeded LCG draw (bit-reproducible, G1-compatible), prompt/keystone rows pinned via the existing `select_evict(pinned)` mask, per-head independent draws (the paper's superadditivity result makes shared draws a different, weaker policy — do not conflate). Matched budget + buffer, same fixture. **Non-vacuity by construction:** the paper's passcode regime predicts the arm collapses at cap=16 (needle-at-depth); a TIE at cap=16 would instead refute signal value on this workload and mass_age's remaining case is protection alone — a demote-the-loser input, recorded either way. *(LANDED 2026-09-06: `rand` + `rand_keystone` arms, dedicated draw stream per trial (fixture bit-identical across arms — the six original cells reproduce exactly), per-eviction uniform tickets (paper Eq. 3), per-head independence structural. Non-vacuity: collapse CONFIRMED (rand@16 = 7/384 floor-class); signal value CONFIRMED at every regime cap — mass_age 5.0×/4.8×/3.8× the null at 32/48/64. The demote-the-loser branch did NOT fire. Bonus inversion recorded: rand 7/384 BEATS mass_age 0/384 at cap=16 — the null's geometric survival is a second extreme-pressure loss for mass_age.)*
- [x] **T3.7** Protection factorial: run `mass_age`, `ega_energy`, and the new null arm each in ±prompt-pin form; record per-arm keystone-survival fraction (the paper's keep-log statistic — the attribute-quality-to-protection-vs-signal diagnostic). Update Bench 697 with the extended tables and re-state the G8 verdict against the controlled null. *(LANDED 2026-09-06: `mass_age_keystone` + `ega_energy_keystone` arms; pin-honored gate PASS (all +pin arms 384/384 every cap — mask verified on real traffic); protection-alone ceiling reproduced (every pinned arm = 100% — the paper's Table-2 shape); keep-log ≡ recall in this fixture (one immutable row per needle) — recorded as the collapse note. G8 re-stated: g8_all FAIL (the 3 pre-existing misses) AND mass_age > rand at 32/48/64 PASS.)*
- [x] **T3.8** Record the round-cost axis in the bench doc: our score update is 1.22 ns/row CPU (G2) but the paper's +32–43% serving margin lives where scoring is a kernel pass over paged state — i.e. T4.1 (mass-byproduct kernel, Issue 836). Registered alternative: **if deep-needle recall is not load-bearing on our serving workloads, skip the kernel and ship the null policy** (zero kernel work; the `select_evict` API already supports it). *(LANDED 2026-09-06: recorded in Bench 697 §T3.8 — 1.22 ns/row (4090) / 1.78 ns/row (M3) both ≥5× under budget; the registered alternative is now a MEASURED cell: rand_keystone = 100% at every cap with zero scoring work, rand unpinned = 1.8–26% — the null ships only with a structural keystone oracle.)*
- [x] **T3.9** Fold the null arm + protection factorial into the standing promotion gate: any future lossy KV policy promoted to default must beat `random_prompt_pin` at matched budget AND matched protection (extends `runaway_gate`'s promotion rule — the gate side of Research 531). *(LANDED 2026-09-06: `kv_eviction::{PolicyControl, beats_random_prompt_pin}` — strict bar (equal recall hands the slot to the cheaper null), NaN fail-closed; 3 unit tests; standing rule = `runaway_gate` ∧ `beats_random_prompt_pin` ∧ the protection factorial.)*
