# Issue 969 — routing-statistics diagnostic suite (Nestor App C.6) on kimi_k3's shipped MoE + reflex `pick_domain`

**Status:** Open — 2026-10-27. PoC/proof task from Research 639 (arXiv:2610.31818, Nestor).
**Claimed:** zed-amo-969-router-stats, 1793120641, expires 2026-10-29T17:04:01Z

## What

Port the paper's App C.6 routing-statistics methodology as a zero-training, offline diagnostic over FROZEN weights + a corpus, with a liveness canary on a known load-bearing CONTROL. Four statistics, per router/layer:

1. **Effective choices** — `exp(H(p))/E` of the aggregate assignment distribution (load balance).
2. **Corpus MI** — mutual information (bits) between a corpus label and the assignment.
3. **Assignment drift** — % of assignments that change under a perturbation (for kimi_k3: router-weight noise or a LoRA-adapted variant; for pick_domain: corpus evolution between snapshots).
4. **Shuffled/random counterfactuals** — re-route each token/query with another's (shuffled) or a random equal-size (random) assignment at unchanged load; measure the output delta. **Interpretation rule:** a near-zero shuffle cost is a legitimate FINDING ("this router is not load-bearing") on any measured subject — it is a REFUSAL only on the CONTROL (below), where a near-zero delta means the instrument itself is blind.

## Consumers

- **kimi_k3** (shipped forward + real trained checkpoint on disk: `data/kimi-k3-0.40b/model.safetensors`, 1.5 GB; feature `kimi_k3_loader`; skip loud without it): `MoeConfig::kimi_k3_0_40b()` — 8 routed experts top-2 + 1 always-on shared expert, sigmoid router. **Scope caveat: K3 routes at ONE level** (flat top-2; the shared expert is never routed), so the paper's two-level column-vs-expert MI split is NOT runnable here — the answerable question is the single-level one: **which regime does K3's one routed level live in** — corpus-DEPENDENT (like Nestor's expert level, 0.28 bits) or corpus-independent-but-load-bearing (like Nestor's column level, 0.05 bits but 0.4–0.74 nats when shuffled)? **Second caveat: the owner rates this released 0.4B checkpoint as test-arch quality (standing note: "Kimi-K3 0.4B is dumb, use for test arch only") — routing statistics from it describe THAT checkpoint's router; the verdict row is never evidence about Nestor, MoE in general, or production-MoE regimes.**
- **reflex `pick_domain`** (corpus router, `riir-reflex/src/engine.rs` path): shuffle the routed domain per query → top-1 drop = how load-bearing domain routing is. This lane IS two-level (`pick_domain` → per-span), so the paper's per-level MI split applies HERE. **This lane is also the liveness CONTROL**: its seated-suite top-1 drop from domain shuffling is expected to be large; a near-zero drop on this control is an instrument failure — refuse the run loud, never print a green zero.

## Why not covered by existing instruments (signal-diff)

- `benches/bench_015_swe_trajectory_perturbation_sensitivity.rs::perturb_moe` perturbs router WEIGHTS and measures output sensitivity — not assignment STRUCTURE (balance / corpus dependence / load-bearingness under reassignment).
- Research 059 D1 (Raven slot overlap) measures adjacent-token temporal locality — a different question.
- Nestor App C.6's own stats are standard information theory; the delta here is the FUSION (applied to our shipped routers + the counterfactual as a standing liveness gate on a control), not the math.

## Inputs (named)

- **T1 checkpoint:** `data/kimi-k3-0.40b/model.safetensors` via `load_kimi_k3` (the `KIMI_K3_MODEL_DIR` default the kimi_k3 benches already use; bench_012's real-vs-random posture is the pattern).
- **T2 labeled corpus:** the reflex seated dataset suites (`riir-reflex` `.raw/datasets` — ag_news / banking77 / emotion / sst5 are distinct labeled domains); suite label = the "corpus label" for the MI statistic. Same N sequences per suite, seeded order, byte-stable.

## Tasks

- [ ] **T1** — kimi_k3 MoE assignment capture: forward N eval sequences through the real checkpoint, record per-layer routed-expert assignments (feature `kimi_k3_loader`; skip loud without weights). **Provenance to record in the T1 row:** `data/kimi-k3-0.40b/` carries the UPSTREAM Moonshot release layout (`config.json` + `modeling_kimi_k3.py` + `tiktoken.model`, file dated 2026-08-01 — consistent with R447's "K3 weights drop 2026-07-27"), NOT a C13/C14 training artifact; T1 records the source run/step (upstream release) beside the stats so "undertrained router" reads against the right baseline.
- [ ] **T2** — the four statistics + the shuffled/random counterfactual output delta, on the named checkpoint + corpus; golden-serialize the stats (seeded corpus, byte-stable).
- [ ] **T3** — reflex `pick_domain` arm: assignment capture over the seated suites + shuffled-domain top-1 delta on the frozen read; this arm doubles as the liveness CONTROL.
- [ ] **T4** — verdict row vs the paper's Table 9/10 shapes: K3's single routed level — which regime? `pick_domain` — per-level MI split + load-bearingness. Both rows carry the two scope caveats above.

## Gate

G1: the CONTROL arm (`pick_domain` seated suites) must show a large shuffle delta — near-zero there refuses the run loud (instrument-blind). K3's own delta is REPORTED whatever its size (a near-zero is a finding, not a failure). Stats byte-stable across two runs. Report-only instrument; no serving-path change, no feature promotion.
