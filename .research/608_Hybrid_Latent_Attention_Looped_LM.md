# Research 608: Hybrid Latent Attention for Looped Language Models (608)

> **Source:** [Hybrid Latent Attention for Looped Language Models](https://arxiv.org/abs/2610.07940) — Yuhan Chen, Siyuan Zhang, Nan Wang, Feiyang Kang, Ruoxi Jia (Virginia Tech + independent), 6 Oct 2026
> **Date:** distilled 2026-10-08
> **Status:** GAIN — two filings this session: `.issues/926_mla_decode_absorption.md` (PRIMARY — a live-path defect the paper's headline measurement indicts) + `riir-train/.plans/447_ouro_hla_latent_kv_uptrain.md` (SECONDARY — option-creating training plan, status declared there)
> **Classification:** Public (engine primitives only; no game/chain/shard IP)
> **Related Research:** 028 (Higher-order Linear Attention — NAME COLLISION, different paper/group: Zhang/Qin/Wang/Gu Princeton-UCLA vs this paper's Chen/Zhang/Wang/Kang/Jia VT), 073 (LT2 + the looped-LM family verdict stack: Ouro 2510.25741, MoR, SMELT, "Done Right", Loopie, Full-bandwidth), 378 (HOLA — the "exact-hot + compact-cold" hybrid pattern on the GDN backbone), 589 (already flags FlashLoop 2609.29812 as occupying the training-free looped-KV slot), 327/330 (MLA math distillation + actual-model divergence — the served latent-attention prior art), 116 (LLM Sleep — cited by the model-based advocate as an Ouro window-eviction floor; **that attribution was WRONG, corrected here: 116 is sleep-consolidation, no window-eviction measurement lives there**)
> **Related Plans:** 057 (Higher-order LA, shipped), 108 (LT2 `forward_looped`, shipped), 324 (riir-train looped continued-training — DEFINITIVE FAIL on an unconverged base; postmortem asks for a converged base — the standing consumer for the secondary plan), 136 (Training-Free Loop), 926-issue (this note's PRIMARY output)

**Novelty claim: NONE.** Q1 does not apply — nothing here claims a novel primitive. Absorbed MLA is published (DeepSeek-V2; TransMLA arXiv:2502.07864; this paper's §2), and the latent+window architecture is the paper's own. The session's outputs are (a) an **engineering defect finding** on our shipped path, externally evidenced, and (b) routing/plan judgments. This note records both and the Path 0 inventory.

---

## TL;DR

Looped LMs (same layer stack applied T times, T=4 for Ouro) multiply the KV cache by T — 768 KiB/token on Ouro-1.4B, five 16K sequences per 80 GB GPU. The paper stores each older token as a compact latent `c_j = Σ_{t=2..T} E_t h_{j,t}` (rank 512+512) plus a smaller loop-1 latent (rank 256 — loop 1's K/V are the least predictable from the final hidden state, ridge R² 0.66 vs 0.93 for loop 3), keeps EXACT K/V in a W=128 sliding window, and each loop reads the latent DIRECTLY through per-loop maps `A_t` (query→latent space), `B_t` (latent value sum→head dim) — **no K/V reconstruction**. Key-latent pairs carry one rotary frequency each, so the write-time rotation commutes with same-frequency linear combinations (TransMLA's trick extended across loops). Results: 10.7× cache shrink, 4.0–8.8× concurrent sequences, 2.5× decode @1K → 7.4× @16K, ≥97% accuracy retained with the base FROZEN (uptrain only the added maps, 0.35B params, 1200 steps), on-par after SFT.

**The headline empirical law, and why it matters here:** the reconstruct-based predecessor LLA (per-step K/V rebuild from latents) runs at **7–18% of full-cache throughput**; the direct latent read is **28–59× faster than LLA**. Absorption beats reconstruction by more than quality ever will.

**The finding:** our shipped MLA decode path (`crates/katgpt-attn/src/mla.rs`, Step 6) is a reconstruct path — per head, per cached token, per step it recomputes `k_c_j = W_UK[head]·c_kv_j` and `v_c_j = W_UV[head]·c_kv_j`, and its own comment names weight absorption as deferred future work ("For Phase 2 (no weight absorption), we recompute per-token... Phase 6 may cache the up-projected k_c/v_c or use weight absorption"). The paper's law + FLOP arithmetic (§3 of issue 926: ~60× fewer score/value FLOPs at seq 4K on the Kimi-K3-0.40B config) make this the session's PRIMARY actionable item → `.issues/926_mla_decode_absorption.md`.

---

## 1. Path 0 inventory (training-target decomposition)

| # | Component | Track | Closed-form? | Disposition |
|---|---|---|---|---|
| 1 | **MLA decode absorption** (read latent directly; `q_abs = W_UKᵀ q_c` per step; `o_h = W_UV·(Σ_j w_j c_kv_j)`) | (a) modelless — pure algebra | YES — exact in ℝ, fp-reassociation only | **Issue 926** (katgpt-rs, live path) |
| 2 | Looped latent cache (E_t/A_t/B_t + W window + LSE-merged softmax) | (c) model-based — maps are irreducibly trained for the 0.006-nat quality; NO frozen-checkpoint analog (a frozen looped model has never seen latent reads) | Init is closed-form (per-frequency PCA, exact at full rank; KL≈0.29 nats at rank 512 untrained vs 0.006 uptrained) | **Plan 447** (riir-train, SECONDARY) |
| 3 | Per-frequency PCA init (exact at full rank — a free bit-exact canary for any implementation) | (a) | YES | Inside Plan 447 (init arm) |
| 4 | R² rank/window allocator (loop-1 separate latent; joint rank-512 covers loops 2–4 at 0.79/0.89 vs 0.80/0.93 single-loop) | (a) — offline normal equations → schedule | YES | Inside Plan 447 (calibration tool); modest |
| 5 | Distance-conditional divergence meter (0.040–0.049 nats per unit attention at distance 1–127 vs 0.009 beyond 1024; recent tokens = 26.6% of attention mass but 48.7% of divergence) | (b) self-adaptive — measured spectrum → sigmoid-gated window drive | YES (measurement) | Recorded only — no live consumer for an adaptive-W lane today; corroborates the workspace lossy-surface rule (per-family conditional retention > aggregates) |
| 6 | Rotary one-frequency commutation (write-time rotation; `M_f (K ρ(jθ)ᵀ) = (M_f K) ρ(jθ)ᵀ`) | (a) | YES — algebra | Inside Plan 447; only needed if the latent lane is built |
| 7 | Anti-reconstruct design law (7–18% vs 28–59×) | (a) | ordering law | Cited in Issue 926; matches the house anti-pattern "never reconstruct from the embedding — read the latent directly" |
| 8 | Uptraining recipe (AdamW β=(0.9,0.95), cosine, PoSE position-shift curriculum, phase-2 16K docs; KL+rel-MSE self-distill vs frozen teacher) | (c) | recipe | Plan 447 |
| 9 | Attn-KL + rel-MSE per-layer/per-loop gate | (c)/(a) — no training needed to ADOPT as an eval gate | YES (eval-only) | Plan 447 rider (R2); sharper than weight-space fidelity probes — different signal, complementary |
| 10 | Two-pass lazy-cache fine-tuning (pass 1 writes latents under exact attention, pass 2 trains under the decode rule; repetition extends the exact range by W+1 tokens) | (c) | recipe | Plan 447 (banked; general recipe for training against lazily-completed state) |

## 2. Three-track panel (mandatory — the classification touches training)

Both advocates spawned in one parallel batch; their briefs are summarized; every load-bearing claim was re-verified before entering this note.

**No-GD advocate (tracks a+b):** 14 of 15 extractions ship without gradient descent; the paper's ONLY GD step is the 1200-step map refinement, which is regression-shaped against a frozen teacher (recorded teacher signals → ridge/normal-equations substitute; a Plan-025-class deterministic overlay, owning the 0.29-nat untrained envelope as worst case). Ranked absorb-form MLA decode #1 ("same math, different association — strictly fewer FLOPs and bandwidth; the G1 bar is ulp-tight, no quality debate exists"). Also proposed: the ridge-vs-teacher closed-form maps arm (adopted into Plan 447 as a comparison arm), the full-rank bit-exact canary, the sigmoid-gated adaptive-W form for track (b).

**Model-based advocate (track c): FILE, not family-redirect.** Every input of the prior looped-family redirects has changed: (1) cost class collapsed ~100× (maps-only, frozen base, 1200 steps ≈ 15–30 GPU-h vs 105-GPU-h from-scratch-class runs); (2) the "no looped model is served" disqualifier is what this paper's entire purpose is to remove (10.7× cache); (3) the receiving infrastructure is warm — `riir-train-engine/src/kimi_k3_ouro.rs` ships the looped trainer, and Plan 324's postmortem EXPLICITLY asks for a re-run on a converged base (verified: "needs a more converged base", "exit-gate collapse... on an unconverged base"). Ouro-1.4B/2.6B checkpoints are open-sourced (reported Apache-2.0 — pin repo+sha256+license in `data/BASE_MODELS.md` at plan time). **Correction applied:** the advocate's "Research 116 measured window eviction on Ouro" claim FAILED verification (116 = LLM Sleep consolidation); the Floor-B training-free floor stands on the paper's own single-loop-sharing collapse numbers instead, without the R116 citation.

**Coordinator merges:** the panels agree on the split — absorption is the live-path PRIMARY (modelless, algebra, measurable now); the looped latent lane is the SECONDARY training plan (option-creating, standing postmortem consumer, small cost, no chartered serving consumer — its Status line says so).

## 3. The defect, precisely (Issue 926's evidence)

`crates/katgpt-attn/src/mla.rs` `mla_forward_token` Step 6, per head h, per cached token j, per decode step:
- `k_c_j = W_UK[h]·c_kv_j` — a `d_h×d_c` matvec (8,192 FLOPs at d_h=64, d_c=128)
- score = `(q_c·k_c_j + q_r·k_r_j)·scale`
- `v_c_j_h = W_UV[h]·c_kv_j` — another 8,192 FLOPs, then a scaled axpy

Absorbed form: `q_abs_h = W_UK[h]ᵀ·q_c_h` ONCE per step per head (8,192 FLOPs, amortized over the whole sequence); score content part = `q_abs_h·c_kv_j` (a d_c dot); values accumulate in latent space `s_h = Σ_j w_j·c_kv_j` (a d_c axpy per token) then ONE `o_h = W_UV[h]·s_h` per step per head. At seq=4096, n_h=16: ~1.09 GFLOPs/step of up-projection work → ~0.017 GFLOPs absorbed (~60×). The k_r rope dot, the output gate (applies AFTER attention assembly), and `use_nope` structure are all unchanged. Known costs: fp reassociation (the existing `mla_g1_spec_match.rs` tolerance 1e-4 is the bar), and the reconstruct path STAYS as the spec-match reference + kill-switch. `dash_attn/flashmemory_sparse.rs::mla_forward_token_flashmemory` has the same reconstruct shape for its selected blocks — secondary scope, same fix.

## 4. Fusion (what paper × shipped produces that neither alone has)

- **Anti-reconstruct law × our latent paths:** the paper quantifies (28–59×) what our house anti-pattern asserts qualitatively ("never reconstruct from the embedding; read the latent directly"). Issue 926 encodes it as a bench assertion.
- **Hybrid exact/compact pattern family:** Research 378 (HOLA) ships the same shape on the GDN backbone (exact surprise-evicted KV complementing O(1) recurrent state); this paper ships it for looped softmax attention (latent tail + exact window); Lighthouse/SpectralQuant compress within softmax attention. Three backbones, one law: exact where divergence-per-unit-attention is high (near/recency/surprise), compact where it is low.
- **Two-brain reading (game context reframe):** exact window = info brain (recent, exact, deterministic); latent tail = think brain (compressed, stale, gracefully diverging, never reconstructed). The divergence budget is a sigmoid drive over a measured quantity. No new riir-ai mechanism needed — the belief-kernel family already lives this pattern; recorded as corroboration.
- **Healer context reframe (priority #2 check):** no consumer. The healer's retrieval/trajectory/corpus surfaces carry no per-loop KV cache; the paper's mechanisms have no healer-shaped instantiation. Recorded as an explicit negative so the next sweep doesn't re-derive it.

## 5. Closest cousins and why they don't kill the (non-novelty) outputs

1. **LLA (O'Neill & Reid, arXiv:2607.15456)** — the direct predecessor; reconstructs per-loop K/V from a low-rank latent each step. Not previously tracked in the workspace (this note is its first internal record). It does not kill anything: it is the paper's own baseline and the measured loser (7–18% throughput).
2. **FlashLoop (arXiv:2609.29812)** — training-free looped KV reduction (token sparsity + residual quantization). Already flagged in Research 589 as occupying the training-free looped-KV slot. Complementary to the latent lane (the paper says so); no conflict with Issue 926 (kimi MLA is not looped).
3. **Research 28 Higher-order Linear Attention + Plan 057** — NAME COLLISION ONLY (fourth in-stack HLA: riir-engine `hla/`, AHLA, `HlaCacheProxy`, this paper). Our HLA replaces softmax attention with O(1) recurrent state (needs from-scratch training); this paper PRESERVES softmax attention and compresses its cache (frozen-base uptrain). Different points in the design space; the Plan-447 lane must be named to grep cleanly (`hla_latent_kv` + a grep-warning comment).
4. **MoR recursion-wise KV sharing / Ouro single-loop sharing** — both previously verdicted (Research 73 stack); this paper MEASURES the single-loop-sharing collapse (MATH500 45.9 vs 75.9 at 2.7× HLA's cache size) — an external confirmation of the family's redirect logic, now with numbers.

## 6. Per-stack ledger (katgpt-rs)

- **Slot:** attention/KV. New primitive: absorbed-MLA decode (`mla_absorbed`), feature-flagged, GOAT-gated (G1 spec-match ≤1e-4 vs the reconstruct reference, G2 decode latency × seq {1K,4K,16K} `--release` + bench_preflight provenance, G3 no-regression, G4 alloc-free). Promote to default only on measured gain; reconstruct stays as reference + kill-switch. The looped latent lane is riir-train-side (Plan 447) and lands in katgpt-rs/riir-infer only as a serving spec handoff — not this session.
- **Demote rule:** if absorption wins G2, the reconstruct path is demoted to reference-only (it is the algebraic spec the absorbed form is tested against).

## 7. Priority
## Priority

- **P1:** Issue 926 (MLA absorption) — live path, algebra-only, externally evidenced, cheap to gate; kimi-lane-scoped (0.40B is test-arch per owner rule; league models don't run MLA), hence P1 not P0.
- **P1:** Plan 447 Phase 1 (base pin + PCA init + teacher harness) — unblocks the training run without committing GPU yet.
- **P2:** Plan 447 Phase 2 (maps run + retention eval) — ~15–30 GPU-h on the 4090; owner-gated spend.
- **P3:** LSE-merged window+latent kernel; adaptive-W drive; flashmemory_sparse absorption rider.

## PASS-Redirects (synthesis)

> **PASS-Redirects (synthesis):** O'Neill & Reid [arXiv:2607.15456 "Looped Latent Attention: Cross-loop KV compression for looped transformers"] — the reconstruct-based predecessor of the paper distilled in this note; internally tracked HERE for the first time (no prior note greps this ID); its measured fate (7–18% of full-cache throughput; ≤ HLA accuracy at 2.7× cache) is the reason the direct-read design won; no separate note warranted.

> **PASS-Redirects (synthesis):** Meng et al. [arXiv:2502.07864 "TransMLA: Multi-head latent attention is all you need"] — the frequency-aligned cross-head key combination this paper extends across loops; consumed as prior art inside the Path 0 table (row 6) and Plan 447's init arm; no separate note (absorbed-MLA lineage already lives in Research 327/330 + the kimi_k3 serving path).
