# Issue 756 — an unterminated fenced code block swallows the rest of its file, and the dangling fence is not where the defect is

STATUS: RESOLVED 2026-09-12 (katgpt-rs half) — 12 files repaired, `markdown_fence_gate.py` landed in the docs gate (17 checks), both arms canaried. 7 findings remain in riir-ai (5) and riir-train (2), reported below for their owners

## The defect

Everything after an unterminated ` ``` ` renders as code — headings, tables,
nav footers, entire subsequent sections. Measured 2026-09-12 with
`skill_repo_set_gate.fenced_blocks()` over all 19 contract repos:

**19 files, 5048 tracked `.md`, 611 lines swallowed.**

The worst was this repo's own `.plans/048_research_audit_fixes.md`: a
```bibtex block opened under "Research Citations" at line 456 and never
closed, turning the following **146** lines — a whole second document section,
"Research Audit Results (Plan 048)" — into a code listing.

It is also a parse hazard, and that is why it is a gate rather than a cleanup.
A scanner that toggles state on every fence line mis-phases permanently from
the first unterminated one and thereafter scans the complement. Precedent in
this repo: `.agents/skills/rust-optimize/SKILL.md`'s unclosed ` ```text `
swallowed `skill_repo_set_gate.py`'s own first canary, and the canary
"passing" was the only symptom.

## The reported line is NOT the defect — three shapes, all measured

A single stray fence inverts the pairing of **every fence after it**, so the
dangling fence the scanner reports may be a perfectly good CLOSER whose partner
was consumed upstream. All three shapes occur in the workspace:

| shape | example | repair |
|---|---|---|
| **missing closer** | `.plans/048` — ```bibtex under a "Citations" heading, file ends inside it | insert the closer |
| **stray fence** | `riir-ai/.plans/094:214` — a second ` ``` ` immediately after a correctly closed ```rust block | delete the line |
| **missing opener** | `riir-train/.docs/…/training_handoff_4090.md:1375` — a panic trace meant to be fenced, closer present, opener absent | insert the opener |

The discriminator is the **first non-blank body line** after the dangling
fence: code means a closer is missing, prose means the fence is an orphan.

⛔ Two classifiers were wrong before one was right, both in the direction that
would have caused a bad edit:

1. **"the block body contains markdown headings ⇒ mis-paired"** — `#` starts a
   comment in toml, sh, python and text. `# Cargo.toml — separate game from
   language concerns` inside a ```toml block and `# Standard: use only final
   layer` inside a ```text block both read as `<h1>`. Under this rule
   `.plans/040` and `.research/068` looked like they had upstream phase breaks;
   read by hand, both were correctly paired. Repairing at the "located" site
   would have introduced the defect being repaired.
2. **"the dangling body contains prose ⇒ the fence is a stray"** — a *missing
   closer* swallows downstream prose too, which is the entire complaint. It
   cannot separate the shapes. `.plans/048` was misclassified STRAY (delete)
   when it needed a closer inserted 25 lines later.

Only a signal that does not occur inside code — a bare `---`, a `[←` nav link,
a `**Bold` line start — separates them, and even then the body's first line is
what decides the repair.

## Repairs — 12 in this repo

10 × append a closer at EOF (`.benchmarks/020`, `021`, `.contexts/optimization`,
`.plans/040`, `.plans/044`, `.research/015`, `016`, `017`, `018`, `043` — all
bibtex/sh blocks under a trailing "Citation" or "Run Command" heading), 1 ×
insert a closer mid-file (`.plans/048` @482, before the `---`), 1 × delete an
orphan fence (`.research/068` @499, between a table and an `---`).

Each anchor was asserted before the write (the preceding and following lines
matched literally, else abort) and the files were edited as **bytes** — they
end without a trailing newline, and text-mode writes rewrite line endings.

Verified: `fenced_blocks()` reports **0** dangling fences over 1517 tracked
`.md`, and 0 closed blocks contain a prose marker (no mis-pairing left).

## The gate — `scripts/markdown_fence_gate.py`

In the docs gate (now **17** checks). Walks tracked `*.md`, exits 1 on an
unterminated fence, **2** if the walk falls below `MIN_FILES = 800` — a ceiling
over zero files is green for the wrong reason. It imports `fenced_blocks` from
`skill_repo_set_gate.py` rather than re-deriving it (Issue 755's lesson: one
implementation of a subtle rule beats two that agree today).

Canaried both arms, because a gate that cannot fail certifies nothing:

| arm | result |
|---|---|
| planted unterminated fence | exit **1**, names the file and line |
| floor raised above the walk | exit **2**, INSTRUMENT |
| restored | exit **0**, 1517 files |

The gate's own landing was verified by an unrelated gate: adding the AGENTS.md
row citing "Issue 756" **red the citation gate**, because 756 was not yet
allocated locally — exactly the rebinding hazard Issue 749 exists to catch,
demonstrated on the commit that documents it.

## Remaining — 7 findings in two sibling repos, NOT repaired here

Reported rather than edited: both repos have active concurrent sessions, and
these are their documents to fix.

| repo | file | line | shape |
|---|---|---|---|
| riir-ai | `.research/005_Secure_Speculative_MMORPG_Engine.md` | 746 | stray fence after a prose paragraph (109 swallowed) |
| riir-ai | `.plans/094_gpu_forward_pass_uniform_race_fix.md` | 214 | stray — duplicate closer after a ```rust block (39) |
| riir-ai | `.plans/052_g_zero_bomber_selfplay.md` | 174 | missing closer, "File Map" tree (24) |
| riir-ai | `.plans/172_multi_agent_job_matrix.md` | 282 | missing closer, "File Structure" tree (21) |
| riir-ai | `.research/002_Geometric_Calculator_Fourier_Mechanism.md` | 927 | missing closer, mapping table (12) |
| riir-train | `.docs/01_orientation/training_handoff_4090.md` | 1375 | missing OPENER before the panic trace at 1373 (114) |
| riir-train | `.plans/332_edged_training_goat_unification.md` | 1150 | missing OPENER before the box diagram (14) |

The two riir-train rows are the missing-opener shape and are the reason the
caveat above is stated so strongly: their reported line is a correct closer.
