# Bench 927 — Issue 929 T1.2 checkpoint run: the loop-alignment probe on the real Kimi-K3-0.40B (verdict: UNDECIDABLE — 0% task accuracy at every elicitation posture)

**Status:** COMPLETE — instrument landed, checkpoint leg executed, verdict UNDECIDABLE (not a T1.4 kill: AUROC is undefined, never computed). T1.3/T1.4 stay armed on a capable checkpoint. [Issue 929](../.issues/929_loop_alignment_probe_decode_reinject_poc.md) · Research 614 · arXiv:2607.00341.
**Date:** 2026-10-10 · **Box:** shikuwa 4090/13700K, CPU-only single-thread run (safe beside the plan437 GPU trainer — no GPU exclusivity consumed) · **Commit:** this one.

## What landed

- `examples/kimi_k3_loop_alignment_probe.rs` (root, `required-features = ["kimi_k3_loader", "loop_alignment_probe"]`): the runner the issue named "next unit". Prefill → K caller-side Ouro iterations (`kimi_k3_forward_token_hidden`, hidden re-injected — each step is one KDA update + MLA append, no token consumed; the Plan 324 A4 backward-twin path) → per-loop `probe_alignment` (cos + margin, zero extra GEMV — the forward's logits ARE `W·H^(k)`) → loop-refined greedy decode → labels (full-match primary / first-token / pre-loop control) → `auroc` + stratified bootstrap CI per (k, signal) → the T1.4 kill-bar verdict line. Env-tunable posture incl. elicitation levers `K929_BOS` + `K929_EXEMPLARS` (solved in-context exemplars, excluded from eval).
- Untied-head precondition asserted at boot (`lm_head` [163840 × 1024] row count) — the probe reads the TRUE readout operator (issue T1.2 precondition, resolved).
- `clippy -D warnings` clean at the feature union; example compiles to nothing without features; `[[example]]` row in the same change.

## The runs (smoke postures, 16-item fixture, seed 0x929)

| posture | L | items | full-match | first-token | verdict row |
|---|---|---|---|---|---|
| bare (no BOS, 0-shot) | 4 | 16 | **0/16** | 0/16 | UNDECIDABLE (pos=0) |
| BOS=1 | 4 | 16 | — (superseded) | — | same degenerate output family |
| BOS=1 + 2-shot exemplars | 4 | 14 | **0/14** | 0/14 | UNDECIDABLE (pos=0) |

Generations are memorized-SFT regurgitations ("a story the name is Inigo Mont…", "…Darth Plagueis", "a bee should be able to fly"), near-identical across postures while still varying per item — the prompt reaches the model; the model ignores the synthetic fact context. Kimi-K3-0.40B is the arch-test checkpoint (owner: "use for test arch only") and it cannot do the two-hop composition task even one-hop, at any elicitation tried.

## The measurement that DID read something (secondary observation, all-neg data)

- `cos(H, W[v̂])` sits at **0.29–0.37 across k** — the paper's reported misalignment range (0.27–0.33) reproduces in MAGNITUDE on this non-loop-trained checkpoint.
- The loops sharpen decode confidence **margin 1.1 → 2.9–4.4 (peak k=2–3) → decay by k=4** while cos stays ~0.32–0.35 — loop iterations act as a confidence amplifier that does NOT improve state-vs-readout alignment. ID vs OOD trajectory split is small (~0.01–0.02 cos).
- No AUROC row exists anywhere: `pos=0` at every posture ⇒ every AUROC is undefined (the instrument prints UNDECIDABLE and refuses to fold that into any pass column — by construction).

## Verdict

**UNDECIDABLE — not the T1.4 kill.** The kill bar (AUROC 95% CI LB < 0.6) presupposes both classes; on the only executable checkpoint surface in katgpt-rs (kimi_k3 family; the Bonsai/MiniCPM `forward_looped` leg is arch-impossible per the issue re-scope, and 4B-A2B has no real checkpoint) the task-accuracy axis is dead at 0. The probe question stays **armed**, with two named re-open conditions:

1. **riir-train Plan 451's loop-trained artifact** (the issue's standing reopen trigger) — a checkpoint trained to loop makes both the misalignment premise AND the correctness axis live.
2. Any future kimi_k3-family checkpoint with nonzero task capability on this fixture (Plan 437's training lanes are the candidates).

The full 56-item posture was NOT run: three postures × 0% with elicitation-insensitive outputs is conclusive on the label axis regardless of n; one command re-runs it if wanted (`cargo run --release --features kimi_k3_loader,loop_alignment_probe --example kimi_k3_loop_alignment_probe` with `K929_OUT=<path>` for the JSON record).

## Cross-refs

- Issue 929 (T1.2 checkpoint leg CLOSED landed; T1.3/T1.4 armed) · Research 614 · Benches 847/850/906 (the 0-for-3 loop-intervention ledger this probe was measuring ahead of) · Plan 451 (trained-to-loop lane) · Plan 437 (the trainer holding the GPU while this ran CPU-side).
