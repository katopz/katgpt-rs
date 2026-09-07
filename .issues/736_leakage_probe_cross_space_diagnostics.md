# Issue 736: leakage_probe + cross_space diagnostics — modelless open primitives (Super-GOAT: T1–T3 landed)

**Status:** IMPLEMENTED (T4 clean + T1–T3 landed 2026-09-07, opt-in `leakage_probe`, gates G1/G1b/G1c/G2 10/10 PASS). Source: arXiv:2505.12540 "Harnessing the Universal Geometry of Embeddings" (vec2vec, NeurIPS 2025).

## T4 — defense-audit novelty deep-search (DONE 2026-09-07)

Searches run: "embedding inversion attack defense detection text embeddings" · "vector database privacy leakage auditing membership inference embeddings" · (attribute-inference-defense search timed out; the two returned cover the landscape).

Landscape verdict — three adjacent classes exist, none is a leak-audit score:
1. **Attack tools** — vec2text (2310.06816), ALGEN few-shot inversion (2502.11308), Transferable Embedding Inversion (2406.10280). Attacks, not audits.
2. **Defense mechanisms** — DPPN (perturb privacy-sensitive neurons), Eguard (AAAI 2026, mutual-information embedding shield). *Change* the embeddings to prevent inversion; they do not *measure* transferability.
3. **Risk taxonomies / membership inference** — FINOS air-governance "Information Leaked to Vector Store", Anderson 2025 retrieval-DB membership inference, industry posts. Describe the risk; no quantitative two-space score.

Pinned claim survives: a **modelless cross-space attribute-transfer leak SCORE** (defender-side audit, no encoder access) is unshipped in the literature and in the workspace (Research 540 §3 repo-grep zero hits). Tier: **Super-GOAT** on the 4-question gate (Q1 no prior art in class · Q2 new defensive-audit capability · Q3 "our vector stores ship with a measured attribute-leak score" · Q4 composes MAG × neuron-db security × SipIt × canon transport). Q1 carries the honest adjacency caveat above.

## Landed (T1+T2+T3 as one module family — they share one kNN kernel)

- [x] T1 — `katgpt-core/src/leakage_probe/` (feature `leakage_probe`, opt-in, zero deps): `probe()` → `LeakReport { attribute_transfer_top1, chance_baseline, lift, alignment_mean_cos, neighborhood_hit_rate, verdict }`; verdict tiers InsufficientAlignment/Low/Elevated/High.
- [x] T2 — cross-space diagnostics folded into the report: `neighborhood_hit_rate` (label-free transport quality) + `alignment_mean_cos`, sharing the probe's kNN kernel (DRY — no MAG feature implication needed).
- [x] T3 — unpaired transport inside `leakage_probe::transport`: deterministic subspace iteration → PCA whitening (shared `linalg::symmetric_eig`) → CSLS-corrected entropic Sinkhorn → orthogonal-Procrustes polar factor; deterministic multi-start (k+2 orientations, best mean pair cosine) against wrong-basin ICP lock-in. The G1c control PINS the honest negative (iid foreign labels → chance) in-test.
- [x] T4 — this search; verdict recorded above.

## Gate results (in-module, `--features leakage_probe`, 10/10 PASS 2026-09-07)

- G1 planted-leak recovery: 4-cluster non-congruent fixture (distinct per-cluster scales — congruent blobs are unpaired-alignment-ambiguous, the same degeneracy that makes the paper's OT baselines fail) → High, top1 ≥ 0.75 vs chance 0.25.
- G1b monotone-in-noise: transfer accuracy non-increasing in observed noise; clean ≥ 0.7.
- G1c honest negative: iid foreign labels → lift ≤ 1.3, Low.
- G2 audit-cadence smoke: n=128/d=32 completes ≪ test budget (release measurement deferred to the E2 adoption bench).
- Validation: `cargo clippy -p katgpt-core --features leakage_probe --lib --all-targets` 0 warnings; default + `--no-default-features --features leakage_probe` both compile; docs_gate 14/14 (feature-count claims bumped 578→579).

## Follow-ups

- [x] T5 — riir-neuron-db Issue 614 E2 adopts the probe as its security test (consumer gate; GOAT bench lands there). **DONE 2026-09-07 (riir-neuron-db `51e2ca1` + `6e14f4d`, [Bench 495](../riir-neuron-db/.benchmarks/495_leakage_audit_goat.md)):** 3/3 gates PASS — planted recovery through the real steal path (top1 0.828 vs chance 0.086, lift 9.64, High), integrity-≠-confidentiality pin, monotone-in-coefficient 0.828 → 0.082 across α 1.0 → 0.05. Release G2 measurement landed there: probe 380 ms @ n=256 768→384 (1.49 ms/row), steal 48 µs, multistart price 2.03×. Probe wired via a TEST-ONLY dev-dep feature (lib closure unchanged). **Fixture lessons fed back (corpus candidates):** (a) ±paired cluster centers make the multistart's best-mean-pair-cosine criterion BLIND between the true and the antipodal basin (both ≈ +1 on their own pseudo-pairs) — measured exact 0.000/0.500 top1 flips across coupling rungs; (b) gain-attenuation against fixed noise only changes cluster TIGHTNESS on unit-norm rows (scale-invariant geometry), so recovery moves the wrong way; (c) the structural fix is 12+ generic independent centers — no permutation re-superimposes the arrangement, margins survive both spaces' whitening, and the ladder is cleanly monotone.
- [x] T6 — architectural guide for the stolen-DB risk quantifier (probe × Plan 391 frozen translator × SipIt endgame) — filed as `riir-neuron-db/.research/308` (the stored-latent security domain owns it). **DONE 2026-09-07:** the guide landed there complete — TL;DR commercial claim, the four-question Super-GOAT verdict, the probe × translator × SipIt connection map, and the freeze/thaw packaging contract; this checkbox was the only stale bit.

## Design lessons (recorded for the heuristic corpus)

1. Congruent clusters are unpaired-alignment-ambiguous — ANY bijection is self-consistent; shape-distinct structure is what makes correspondence identifiable (same mechanism behind vec2vec's OT-baseline failure).
2. Plain-cosine ICP locks onto hubs in isotropic clouds — CSLS correction (Conneau et al. 2018) is the canonical modelless fix.
3. LCG consecutive draws correlate in the high bits — fixture streams must be per-column seeded or the sample covariance tilts off the coordinate axes.

## Refs

`katgpt-rs/.research/540` (feasibility + coverage) · `riir-train/.plans/391` (frozen-translator track) · `riir-neuron-db/.issues/614` (threat + consumer) · MAG Plan 418 · SipIt Plan 561 · CD-LAM Issue 194 · Bench 194 · arXiv:2505.12540
