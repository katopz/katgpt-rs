# Issue 762: ASEntmax P4 stretch — HoldConcentration softmax upgrade, Kamath regime detector, per-head grid table, RoPE cutoff spectrum

**Status:** OPEN — stretch backlog deferred at Issue 747 close-out (2026-09-12); no owner priority yet.
**Origin:** Issue 747 P4 (resolved + removed; full record in git history `.issues/747_asentmax_modelless_mining.md`) · **Research:** [katgpt-rs/.research/549](../.research/549_ASEntmax_Length_Adaptive_Entmax_Attention.md) · **Evidence base:** Bench 713 (+ P1/P2/P3/P0.7 addenda).

The real-model harness landed at Issue 747 P0.7 is reusable for every task
below: `crates/katgpt-attn/examples/asentmax_p07_gen_fixture.rs` (in-repo GGUF
reader + Q2_0_g128 dequant + qwen3 prefill of Ternary-Bonsai-8B, PPL-validated
against llama.cpp), the committed 1.16 MB fixture (1856 routing rows / 64
block-summary streams / oracle masses), and the replay gate
`tests/asentmax_p07_realmodel_regate.rs`.

Priority context from P0.7's measured verdict: real routing σ̂ ≈ 0.1409 — an
order below the σ ≥ 1 over-sparsification regime, and the derived schedule
showed NO modelless quality gain on the real path (needle parity 96.3%/96.3%,
oracle-mass −0.017 scheduled; `asentmax_schedule` stays opt-in). T4.3/T4.4
only earn their keep if a real large-σ surface or longer-context regime
(n > 32 blocks) appears — re-measure before building.

## Tasks

- [ ] **T4.1** `SsmaxMode::HoldConcentration{c,k}` — analytic `θ*(n) = Δ̂/ln((n−k)c/(k(1−c)))` from Lemma 2's softmax side (upgrades shipped SSMax with the exact coefficient).
- [ ] **T4.2** Kamath range-law detector `ρ = Δ̂/(2σ̂√(2 log n))` (Gaussian vs spiked regime) + normalized-entropy dispersion diagnostic `H(p)/log n` (O(s) over support) — `katgpt-core` estimator module beside ssmax.
- [ ] **T4.3** Offline per-head (β,γ) grid sweep (frozen table, freeze/thaw-versioned) — only if the derived γ=−0.5 default passes P0 and a swept fit demonstrably beats it; harvested constants may also arrive from riir-train Plan 396 Ph2.
- [ ] **T4.4** Prop 7/8 RoPE cutoff spectrum — retention must respect periodic re-entry windows (window union, not first cutoff).
