# Doc-Status Auditors — Feature-Gate vs Doc-Label Drift

Status: **living** (last re-verified clean 2026-07-18, Issue 180 session 16)
Origin: `Issue 180` (now removed per noise-reduction rule)
Design record of the 2026-09-01 rewrite (dialect tokenizer, `(package, feature)` reachability closure, `on by transitive default`, sibling cadence): [`sibling_doc_drift_auditors.md`](sibling_doc_drift_auditors.md) (Issue 702, closed + removed)

## What these auditors catch

Two scripts under [`scripts/`](../../scripts/) keep documentation status claims
in sync with what `cargo` actually compiles. Both surface the same drift class
— **doc says X, Cargo.toml says Y** — but on different surfaces:

| Auditor | Surface | What it parses |
|---|---|---|
| [`scripts/bench_doc_audit.py`](../../scripts/bench_doc_audit.py) | `.md` docs (benchmarks, plans, `.docs/`) | Feature-gate status labels: "default-on", "opt-in", "DEFAULT-ON since ...", "Opt-in", etc. near `Feature:` headers. Resolves transitively-enabled features (`sense_lod → slod`, `bom_sampling → micro_belief`, `hybrid_oct_pq → planar_quant → turboquant`, etc.) so the comparison matches what `cargo build` actually links in. |
| [`scripts/cargo_comment_audit.py`](../../scripts/cargo_comment_audit.py) | `Cargo.toml` inline `# ...` comments on feature-definition lines | Hybrid closure strategy: union for default-on claims (cross-crate "via X" / "in root" counts), per-manifest for opt-in claims (root-level opt-in is precise about root). Local-scope overrides for "stays opt-in", "Opt-in in `<crate>`", "NOT in `<crate>` default" patterns. |

## Usage

```bash
python3 scripts/bench_doc_audit.py /git          # .md docs across all 8 repos
python3 scripts/cargo_comment_audit.py /git      # Cargo.toml inline comments

# Or scope to a single repo:
python3 scripts/bench_doc_audit.py /git/katgpt-rs
```

Exit code 0 = clean, 1 = mismatches found.

## Run-after-promotion discipline

**After every feature promotion or demotion, run BOTH auditors.** A promotion
that adds a feature to `default = [...]` (or removes one) is exactly when the
`.md` status label + the Cargo.toml inline comment both need updating — the
auditors catch any miss.

The auditors are static (doc text vs Cargo.toml). They do NOT re-run benchmarks
or rebuild anything. They take seconds to run across all 8 repos.

## Honest limitations (carry-over from Issue 180 sessions 9–16)

- **Static only.** Doc/Cargo drift is detected; semantic correctness of the
  claim itself is not.
- **Two phrasing classes survive the auditor** because runtime-vs-feature-status
  is a judgment call: "transitively default-on" vs "opt-in" vs "default-on" for
  features that exist in the default closure but only via a parent feature.
- **~502 broken `.md → nonexistent file` path refs** across all 8 repos
  (Issue 180 sessions 9–10 + 15 residual) are NOT covered by these auditors.
  That drift class — shell-glob braces, forward-looking plan paths,
  refactor-renamed cross-repo refs — has too poor a signal-to-noise ratio to
  gate. The existing 61 doc-level "historical file paths" annotations cover
  the worst offenders; a future plan could sweep this if a concrete cost
  (broken link in published docs, dev confusion) materializes.

## Lessons baked into the scripts (Issue 180 session 13)

1. **Markdown bolding around colon.** Initial regex
   `^\s*\**\s*Feature(...)?\s*[:\-]` did not allow `**` between `gate` and `:`.
   Real headers like `**Feature gate:**` were silently skipped. Fix: `\**\s*[:\-]`.
2. **Substring matching of "default" inside opt-in phrases.** "default-off",
   "off by default", "opt-in, NOT default-on" all matched the bare substring
   `default` and were misclassified as default-on. Fix: word-boundary regex +
   an explicit opt-in-first check for phrases that contain "default" but mean
   opt-in.
3. **No transitive feature resolution.** Features like `slod` / `bfcf_tree` /
   `sense_composition` / `micro_belief` / `turboquant` / `spec_cost_model` /
   `engram` are not in `default = [...]` directly but are enabled transitively
   via other default-on features. Without transitive resolution, the script
   reported false negatives (docs saying opt-in when the feature IS compiled-in
   by default). Fix: closure walk over the feature graph before classifying.

## Why this lives in `.docs/10_audits/`

The `10_audits/` folder holds point-in-time audits that informed structural
decisions. This doc is **slightly different** — it documents **living tooling**
(re-runnable scripts), not a one-off audit. It lives here because (1) the
auditor pattern was born out of the Issue 180 audit cycle, and (2) there is no
better-fit folder. If a dedicated `scripts/` doc surface is added later, this
file is a candidate to move.

## See also

- `Issue 180` — REMOVED per
  noise-reduction rule; the 16-session audit history (sessions 1–16, including
  parser-bug fixes, transitive-resolution passes, and the canonical "false
  claim about `.tmp_bench_audit.py`" investigation) lived there. This doc
  distills the durable parts: what the auditors catch, how to run them, the
  run-after-promotion discipline, and the residual drift classes accepted.
- [`claim_rubric_audit.md`](claim_rubric_audit.md) — adjacent rubric audit
  (claim evidence ladder vs runtime probes).
