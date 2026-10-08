# Research 611: TWS — Event-Anchored Sigmoid State Windows + Identity-Free Population Tokens

> **Source:** "Neural Data Needs Semantic Tokenization: Behavioral Events as Boundaries of Session-Transferable Tokens" — [arXiv:2610.03001](https://arxiv.org/abs/2610.03001), Sangyoon Bae & Jiook Cha (Seoul National University), 2 Oct 2026, cs.LG, 10 pages.
> **Date:** 2026-10-08
> **Status:** Active — primitive plan filed (Plan 623); riir-ai guide filed (Research 396)
> **Related Research:** 311 (zone affective manifold — the continuous per-zone cousin), 167 (cross-NPC set attention — the per-NPC-direction cousin), 586 (habituation filter — ships `settling_ticks`, the closed-form settle law), 395 (VISTA episodic sighting ring — event-appended memory host)
> **Cross-ref (riir-ai):** Research 396 (the crowd-regime-token guide — the Super-GOAT selling-point half)
> **Classification:** Public

---

## TL;DR

The paper's tokenizer (TWS) segments each trial at **behavioral event boundaries via products of sigmoids** with **settle offsets** (the population settles into its new regime only ~300 ms *after* the event; the transition window belongs to neither state), then summarizes each state's population with **permutation-invariant aggregation — no unit identity at all**. Tokens built this way decode behavior on **never-seen sessions** where per-neuron foundation models (POYO, CEBRA, NEDS — trained on those very sessions) decode at **chance**, and a tokenizer **frozen on mice transfers to macaques and Utah arrays with only a Ridge probe**. Random-init TWS already beats the trained per-neuron baselines; the backbone is interchangeable (≤5% spread); **wrong boundaries lose 62–70% of the signal and 40 epochs of training cannot repair them**.

**Distilled for katgpt-rs (modelless, inference-time):**

1. **The event-anchored two-sided sigmoid state window** — `w(t) = σ((t−(b_prev+Δ_prev))/γ) · σ((b_next−t)/γ)`: a soft partition of the time axis into states delimited by semantic events, with the **settle transition `[b, b+Δ)` excluded from both states**. The paper trains Δ; we **derive it in closed form** from the EMA settling law already shipped as `settling_ticks(β, ε) = ln(1/ε)/ln(1/(1−β))` (katgpt-core `habituation_filter.rs:172`). Zero trained parameters.
2. **The identity-free population state summary** — mean + spread + quantiles over an *unordered* member set (the paper's own ablation: fixed summaries approach the trained encoder on population-wide targets, and random-init already transfers). Pure order statistics + existing PCA-subspace machinery (riir-ai Research 311 computes exactly such crowd subspaces).
3. **The transfer law (design law, not code):** any aggregate intended to be compared across populations (zones, shards, spawn waves, servers) must consume **no member identity** — per-member embeddings make every new population out-of-distribution by construction. Variables split into **Type A** (population-wide; survives identity removal) and **Type B** (member-specific directions; does not, and that is the accepted price).

---

## 1. Paper Core Findings

- **Problem.** Extracellular electrophysiology records a *different neuron set every session*; neuron labels never recur (29,403 IBL session pairs, zero shared labels). Per-neuron tokenizers therefore build a session-specific vocabulary — every new session is out-of-distribution by default, and per-neuron foundation models collapse to chance on held-out sessions even when pretrained *on those sessions*.
- **Unit that transfers.** Population activity evolves on a low-dimensional manifold that persists across neuronal turnover and across animals. The manifold **changes regime at task events** (stimulus onset, movement onset); a *state* = the part of the manifold spanned between two events, and each state carries its own behavioral meaning.
- **TWS mechanics (Algorithm 1).** (i) Soft segmentation: `w1(t)=σ((b_stim−t)/γ)`, `w2(t)=σ((t−τ1)/γ)·σ((b_move−t)/γ)`, `w3(t)=σ((t−τ2)/γ)` where `τ = b + Δ`, Δ trained, shared across all trials/sessions. The transitions `[b_stim, τ1)` and `[b_move, τ2)` are **outside** all three states — the population "settles" into a regime after a delay (learned Δ1 ≈ 292–311 ms post-stimulus; Δ2 ≈ 26–60 ms post-movement). (ii) Per state: weighted population mean per neuron. (iii) One shared encoder + cross-attention with **no positional encoding over neurons** (permutation-invariant by construction, proven + numerically checked to 4e-6) → k_dim=2 tokens per state on a Grassmannian Gr(2, 256) via Gram-Schmidt. (iv) Any mixing backbone (CNN/Transformer/Mamba/MLP within 5%).
- **Results.** Cross-session movement MCC **0.565** vs **0.000** for POYO/CEBRA/NEDS (leave-one-session-out Ridge probe; POYO pretrained on the held-out sessions themselves). Frozen mice-trained TWS + Ridge probe transfers to macaques/Utah arrays: reach-direction MCC **0.232** vs 0.010 for event time alone, beating every baseline including POYO trained on the target dataset. A probe fit on **5 labeled sessions** beats baselines fit on all sessions ("direction consistency" cos 0.82–0.93 across sessions explains it).
- **The boundary law.** Boundaries jittered by 100/200 ms: −7%/−9%. Fixed/random/shuffled boundaries: **−62%/−70%** — and 40 epochs of behavioral training on wrong boundaries stays below random-init TWS with true boundaries. Training does not repair wrong segmentation.
- **Ablation honesty.** Random init: 0.274 movement (architecture alone). Gram-Schmidt removable (<0.01). k_dim 1/2/4 within 0.003. Time-shuffled pretraining transfers *at least as well* (the learned summary, not temporal order, transfers). Pretraining adds session-invariance, not accuracy (Table 25: from-scratch ≥ full pipeline with enough data). Session identity and behavior occupy **orthogonal directions** — deleting session directions leaves decoding intact.
- **Type A / Type B.** Population-wide variables (movement, RT; our analogue: crowd affect, panic waves) survive identity removal by construction; member-specific variables (choice, block; our analogue: individual NPC decisions) do not — measured by a cross-half index (split neurons, decode from identity-free summaries of one half on the other).

## 2. Distillation — Path 0 inventory (modelless decomposition)

| Paper component | Coverage (ships?) | Extraction (modelless?) | Verdict |
|---|---|---|---|
| Sigmoid gate segmentation (Eq. 1) | `fast_sigmoid` ships (katgpt-types/simd); one-sided decay `σ(−λ·Δt)` ships (two-brain confidence decay) | **YES** — two-sided event-anchored product window is pure math; **open primitive** (nothing ships it) |
| Settle offsets Δ (trained) | `settling_ticks(β, ε)` ships (R586 habituation) — the closed-form settle law | **YES** — Δ = `settling_ticks(β, ε)` called directly (no wrapper), where **β = the update rate of the crowd statistic being windowed** (named at wiring time; the PoC measures it as the tick-horizon over which the per-tick manifold axes stop rotating past ε after an event). The paper *trains* Δ1≈15 bins; that learned value IS an EMA settle time |
| Transition exclusion `[b, b+Δ)` | Nothing ships it (all windows are one-sided or fixed) | **YES** — window product makes exclusion automatic |
| Permutation-invariant population summary | PCA-subspace summary ships riir-ai-side (R311, per-zone, continuous); order-statistics summary over unordered members: nothing generic in katgpt-rs | **YES** — mean/spread/quantiles are closed-form; the paper's own Tables 19–20 show fixed summaries carry most of the signal |
| Grassmannian commitment (Gram-Schmidt) | 311's PCA axes are subspace points already | **YES** — and the paper shows it is *not required* (<0.01) |
| Frozen transfer + Ridge probe | Freeze/thaw ships (NeuronShard, MerkleFrozenEnvelope); frozen direction-vector readouts ship (dot-product + sigmoid law) | **YES** — Ridge is closed-form; protocol = **one global Ridge fit on ~5 labeled windows, then frozen for every context — no per-context refit** (the paper fits once and applies to all held-out sessions) |
| Trained cross-attention encoder + pretraining losses (MSE/diversity/geodesic) | No analog | Marginal on the targets we serve — the honest comparison is **fixed per-regime summaries vs trained TWS** (the paper's Tables 19–21), not random-init vs trained: Steinmetz movement 0.475 (Level-over-time) vs 0.581 TWS = **82% captured by a fixed summary**; NLB movement the fixed summary **wins** (0.476 vs 0.392); Steinmetz choice 0.126 vs 0.151; the trained encoder's 2× edge concentrates on reach direction (0.116→0.232), which the paper's own cross-half index scores **Type-B** (index 0.12 — identity-bearing). The IBL movement gap (0.274→0.565) is boundary-clock-confounded: event time alone decodes 0.908 there (Appendix R). Time-shuffled pretraining transfers ≥ intact (Table 21); pretraining adds nothing at data scale (Table 25) | **Discard (auditable, numbers cited).** Reopen trigger: the PoC measures fixed-summary capture on OUR crowd latents; if it lands below **60%** (vs the paper's 72–82% band — 72% = Steinmetz movement Level 0.419 vs TWS 0.581; 82% = Level-over-time 0.475 vs 0.581), the trained summarizer becomes a riir-train item (a small per-domain head, not a foundation model) |
| Backbone mixing | Everything we ship mixes tokens already | Interchangeable per Table 3 | covered |

**Modelless-validable: ALL components the game side needs have analogs or closed-form extractions. No riir-train deferral.**

## 3. Fusion (what paper × cousins produce that none alone can)

**TWS × R311 zone affective manifold × freeze/thaw readout × `settling_ticks`** = **crowd regime tokens**: event-segmented, identity-free zone-crowd state summaries whose readouts transfer across zones, shards, and full spawn turnover. Full selling-point guide: **riir-ai Research 396**. Open primitive half (this note): the window + summary math, Plan 623.

- vs **R311**: 311 computes the crowd manifold *continuously per tick* with *per-zone axes* ("no two zones sharing the same mood axes") — explicitly not comparable across zones. TWS adds the segmentation (states at `ZoneEventType` boundaries), the settle exclusion, and the comparability-by-construction (no NPC identity in the token).
- vs **R167** (set attention): refines *individual* beliefs from peers — the per-NPC direction. TWS outputs *one population token per state*. Orthogonal; composable (167 refines members, TWS summarizes the refined population).
- vs **R586 habituation / temporal_deriv**: EMA/derivative windows anchored at *nothing*; TWS anchors at *events* and excludes the settle window. `settling_ticks` is reused as the Δ derivation.
- vs **R395 episodic ring**: event-appended per-NPC memory; the ring is the natural per-state window query host.
- **DEC view (theoretical depth):** a state is a manifold *segment between boundaries*; DEC's `exterior_derivative` d gives a segment its boundary events, and region mass from boundary flux (`boundary_flux_mass`) computes a state's aggregate from its boundaries — a win in d≤3 (crowd manifolds, per the curse-of-dimensionality caveat; NOT for 64-dim shards).
- **Consumer surfaces (healer/reflex, honest):** corroboration, not new mechanism — riir-refine's `AstChunker` already segments at *syntax events* (boundary-anchored tokenization avant la lettre), and the paper's Table 3 (tokenizer decides, backbone interchangeable; random-init beats trained per-unit models) is external evidence for the corpus-is-the-model / architecture-first posture in riir-reflex. No healer change claimed.

## 4. Signal-diff on cousins (the §3.6 discipline)

- **R311 `zone_affective_manifold`**: consumes a *continuous crowd snapshot*, emits *per-zone PCA axes + eigenvalues* (sign-fixed per-zone for temporal continuity). TWS-primitive consumes *event boundaries + member latents*, emits *state-delimited summaries with no per-zone basis dependence*. Signal diff: **when to aggregate** (event-anchored states vs every tick) and **who can compare** (anyone vs only the zone that owns the axes). One read of 311's §7 confirms: its sign-fixing exists precisely because its axes are *not* comparable across contexts.
- **R167 `set_sigmoid_attention`**: consumes per-NPC queries/keys (member-relative relevance), emits per-NPC refined beliefs. TWS consumes *no member pairing at all*. Different direction (member→member vs population→token).
- **`habituation_filter`**: consumes raw per-channel signal, emits novelty gates. No segmentation, no population, no tokens. Shares only the sigmoid + EMA substrate (which TWS *reuses* for Δ).

## 5. Verdict

**Super-GOAT.** Novelty gate:

1. **No prior art?** Internal: no event-anchored state segmentation of population summaries ships (grepped: event.gated/event.boundary/regime.boundary across all `.research`/`.plans`/`.docs`; closest = 311 continuous, 167 per-NPC, 586 anchorless EMA — all signal-diffed above). External: boundary-aware tokenization exists for time-series *forecasting* (BT-LSM, OpenReview); event segmentation is cognitive science (Zacks EST — the paper itself separates task-defined regime boundaries from observer-internal EST boundaries, Appendix B); crowd emotion work is per-agent contagion (van Haeringen 2023 review; Sekhavat 2026). **No churn-transferable identity-free crowd regime tokens found.**
2. **New behavior class?** Yes — cross-population crowd regime recognition with zero per-population fitting (311's axes are per-zone by construction; nothing ships comparable tokens).
3. **Product selling point?** "Crowd regimes that transfer: a panic wave, a rout, a celebration is the same crowd state in every zone regardless of which NPCs are online — recognized modellessly from game events + population latents, no per-zone fitting."
4. **Force multiplier?** Connects R311 manifold + `ZoneEventType` vocabulary (6 shipped types) + R395 episodic ring + `settling_ticks` + NeuronShard freeze/thaw + sigmoid-gate law + KG-triple emission (state transitions as semantic triples) + DEC boundary operators. ≥2 pillars (zone gating, event spawning, memory, freeze).

MOAT gate: katgpt-rs = base primitive via fusion (sigmoid mechanics + order statistics + settle law — public, no game semantics). riir-ai = Super-GOAT guide (Research 396). Routing clean; no rethink-class content.

**Mandatory outputs (this session):** this note + `riir-ai/.research/396_crowd_regime_tokens_event_state_guide.md` + `katgpt-rs/.plans/623_event_state_window_primitive.md`.

## 6. Honest uncertainties

- **Domain transfer unproven.** The paper's evidence is neural populations in structured trials. Our crowd affect being Type-A-like (surviving NPC-identity removal) is an *assumption* — the PoC must run the paper's cross-half index on real/synthetic crowd latents before any promotion (Research 396 §validation protocol).
- **Event vocabulary noisiness.** The paper's boundaries are clean task events (stimulus/movement onset, per-trial). `ZoneEventType`'s 6 types are coarser and aperiodic; the settle-offset derivation via `settling_ticks` is plausible (Δ1 ≈ an EMA settle time) but β is only *named* here, not yet measured against real crowd settle behavior — the PoC owns that measurement.
- **The event-time caveat.** On the paper's own data, when the boundary *defines* the label (movement/RT), most of the decode is the boundary time, not neural content (Appendix R). Our GOAT gate must include the paper's own control: compare against an event-time-only probe to prove the token carries population content beyond event identity.
