# Issue 966: Modelless prefix-first grid re-selection for the factorized block drafter (the DCOCO-class corrector, zero training)

**Status:** Open — novelty re-search DISCHARGED 2026-10-27 (claim survives; DRelay 2610.01439 = the new learned-corrector prior art, its pool premise supports ours); premise-gate INSTRUMENT landed (`TopKPoolCoverage`, `correlation_budget.rs`) — the live measurement rides the riir-ai Plan 614 / Issue 1011 lane posture (quiet riir-ai + quiet-Metal window owed)
**Lease released:** zed-amo-966-coverage (claimed 2026-10-27, epoch 1793056465) — the instrument unit landed `08922f9d7` same day; the live measurement belongs to the riir-ai Plan 614 / Issue 1011 lane posture, so the slot is released to it.
**Claimed:** zed-m3-966-reselect, 1793064699, expires 2026-10-29T09:00Z

**Pinned claim (novelty-audit form):** Training-free **prefix-first re-selection over a factorized block drafter's own top-K candidate grid**, transition-scored by a static corpus bigram prior — distinguished from LiLiCorr (arXiv:2608.20530) / DCOCO (arXiv:2610.12570) / Domino / DSpark (all LEARNED correctors) and from prompt-lookup / FLy / Training-Free-Loosely-Speculative (draft GENERATION, entropy gating, acceptance loosening — none re-selects inside a drafter's grid) by requiring **zero trained parameters beyond the drafter itself**.

**Origin:** katgpt-rs Research 636 (DCOCO, 2026-10-26) §7 — paper × `bigram_markov.rs` × `dflash.rs`.

## The mechanism

DCOCO's evidence chain, stripped of the trained head:
1. **Coverage premise** (their §3/Tab 18): a parallel drafter's top-16 pool contains the verifier-committed token ~95% at slot 1 / 57–61% at slot 15 while top-1 sits at ~81% / ~one-in-five. The grid has the right tokens; per-slot argmax fails to select them.
2. **Selection law** (their §5.3/Tab 3): under prefix acceptance, prefix-first greedy beats aggregate-path (sum-max) selection on acceptance AND throughput at equal scores.
3. **Weak-signal robustness** (their App B): removing the pair state costs 7.3% accepted length but the corrector still works; a candidate-blind scorer still beats the uncorrected drafter. ⇒ a weak statistical transition prior is a plausible modelless stand-in for the learned Δθ.

In-stack composition (all three substrates ship):
- `crates/katgpt-speculative/src/dflash.rs` — the factorized block drafter; per-slot marginal buffers (`steps × vocab`, step-major).
- `crates/katgpt-speculative/src/bigram_markov.rs` (Issue 659, R316 path 2) — corpus-built top-m=16 successor table, deterministic build, true conditionals over in-vocab support. **m=16 is exactly the DCOCO K scale.**
- The new part: score `S_p(a,b) = ℓ_{p+1}(b) + λ·log P_bigram(b|a)` over the K×K adjacent pairs (K² lookups per adjacent pair, one batched pass over the sparse rows), then the **prefix-first greedy walk** (O(B·K), caller-owned buffer, zero-alloc).

Lossless by construction: the re-selected block still passes through standard speculative verification — re-selection changes WHICH tokens are proposed, never the acceptance rule (unlike loosely-speculative's semantic loosening).

## Tasks

- [-] Re-run the novelty search at execution (training-free grid re-selection / bigram-rescored spec decode; the 2026-10-26 pass found none — FLy / loosely-speculative / per-step-lossy are different classes — but this is a §4 obligation, not a settled fact) — **DISCHARGED 2026-10-27 (zed-amo-966-coverage): the claim survives, prior-art set extended.** NEW closest prior art: **DRelay (arXiv:2610.01439, 2026-10-01, missed by the 10-26 search)** — prefix-aware selective REPAIR of candidate selections before verification (global reader + causal selector over the draft block), the same motivation ("even when the correct token is present in the candidate pool, a single early selection error prevents subsequent predictions from being used") — but it JOINTLY TRAINS the draft backbone and the selector (candidate-support learning + repair objective), i.e. the learned-corrector family (LiLiCorr / DCOCO / Domino / DSpark) gains a member; the training-free distinction holds and sharpens. Adjudicated NOT-killing: CSD (arXiv:2604.13634, ACL 2026 — training-free but acceptance-side rescue, the loosely-speculative class); CURE (arXiv:2608.00531 — inference-time repair-TREE expansion, not grid re-selection; tree-verify is a closed non-goal); NgramRL (arXiv:2610.32881 — n-gram as a draft SOURCE for RL rollouts, not re-selection). No training-free grid re-selection found (arXiv API, 4 queries, 2026-10-27).
- [-] **Premise gate FIRST (shared with the lane):** per-slot top-k coverage of the verifier-committed token on the target posture (extends `correlation_budget.rs` per-depth EMA with a top-k-pool counter). If pool coverage ≈ top-1 coverage on this posture, the whole class is dead here — close this issue with the negative record, no walk built. — **INSTRUMENT HALF LANDED 2026-10-27:** `TopKPoolCoverage` + `rank_of` in `correlation_budget.rs` (exact counts, not EMA — a posture read must be reproducible from the counts; `record(depth, rank)` with `None` counting against the denominator; `top1_rate`/`pool_rate`/`count` accessors; 5 tests). **THE MEASUREMENT ITSELF rides the lane posture** (riir-ai Plan 614 / Issue 1011 watch note: the read that must run BEFORE any corrector arm) — not executed here: riir-ai is mid-flight with the plan-642 sibling lane's WIP and this box carries sibling Metal measurements (Plan 337 zero-contention rule). Unblock: a quiet riir-ai + quiet-Metal window; the consumer wiring is the lane's half-line (`coverage.record(d, rank_of(&topk_pool_d, committed))` per depth after each verify).
- [ ] `grid_reselect` feature flag: `dflash_predict_with` gains top-K retention per slot + the scored-grid walk (S = ℓ(b) + λ·bigram(a,b)); flag-off byte-identical
- [ ] λ calibration: small sweep on the in-stack harness. ⚠ **Scale note:** the bigram term is a log-PROBABILITY while ℓ is a raw logit — the sweep calibrates the SCALE between two differently-scaled quantities, not a shape hyperparameter; record the chosen scale beside λ so a future reader does not misread it
- [ ] **λ=0 equivalence check (free G3 arm):** at λ = 0 the prefix-first walk reduces to per-slot argmax — the walk must reproduce the baseline BIT-FOR-BIT before any λ > 0 claim is read (this certifies the walk itself, independent of the prior)
- [ ] GOAT G1: accepted length / effective tokens per cycle vs the argmax baseline (`correlation_budget` reporting, slot-resolved) — **plus a per-slot bigram HIT-RATE column** (share of the K×K grid pairs found in the top-m table). Sparsity hazard: the table keeps only the top-16 successors per token, so most grid pairs will MISS and get zero bigram contribution — a low hit rate means the negative is about TABLE SPARSITY, not about bigram priors in general; the cheap variants before closing are a per-row backoff and a larger `top_m`
- [ ] GOAT G2: walk + scoring latency budget (O(B·K) lookups; criterion bench, `--release`)
- [ ] GOAT G3: no regression on `dflash` core paths (flag-off bit-identity) + `dd_tree` consumers of the same buffers
- [ ] GOAT G4: zero-alloc walk (canary-armed counting allocator)
- [ ] Honest-negative clause: if G1 shows no gain over argmax at any λ, record the negative (the bigram prior over a large vocab may be too weak — the paper's signal is contextual, ours is marginal) and close; the walk + coverage instrument survive as the reusable parts

## Non-goals

- No training, no fine-tune, no adapter (that is the point; the trained-corrector lane option lives in riir-ai Issue 1011's watch note, owner-gated riir-train)
- No tree-verify composition (closed-negative, riir-ai Issue 1068)
- No claim against the league cells (this is an in-stack harness result; the league drafted lane is riir-ai Plan 614's)
