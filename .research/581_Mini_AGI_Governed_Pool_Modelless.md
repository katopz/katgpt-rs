# 581 — mini-AGI: The Governed Paged Pool (modelless extractions)

**Status:** DISTILLED — all three primitive candidates IMPLEMENTED 2026-09-22 as Issue 873 (closed 2026-09-25, HISTORY.md § Issue 873) opt-in features: A `pool_admission` ([Bench 873](../.benchmarks/873_pool_admission_goat.md), commit `eabd0cb8`), C `dying` ([Bench 874](../.benchmarks/874_dying_goat.md), commit `650faeac`) by the parallel session, B `rate_control` ([Bench 875](../.benchmarks/875_rate_control_goat.md)) landing the same day — a same-day TWIN landing (origin's A/C commits canonical; this session's duplicate A/C work discarded per the Batch-169 precedent, B is unduplicated). B5 consumer A/B open on riir-train Plan 416. Training-track rows: [riir-train Research 456](../../riir-train/.research/456_Mini_AGI_Trunk_LR_Continual_Recipe.md) + [Plan 416](../../riir-train/.plans/416_mini_agi_trunk_lr_plasticity_forgetting_probe.md).

- **Source:** `volotat/mini-AGI` @ `96784b78008a22c4a1ee2c9961640c22137320ef` (MIT, 2026-09-22) — a continual-learning byte-level MoE LM: 540M params (169 experts × 3.15M, growing), trunk = embeddings/attention/routers/halting; experts live as files on disk, paged disk→RAM-LRU→VRAM-working-set (32 resident); batch-1 stream training on 8 GB VRAM; no catastrophic forgetting (trunk LR at 0.1× experts). Read via `.raw/` clone per §0.5; clone removed after commit.
- **Verdict (per track):** training track → **Gain** (Path 0.5 Plan, riir-train 416). Modelless track → **Gain** (3 primitive candidates, Issue 873). **Not Super-GOAT** — Q2/Q3 fail: the demand-paged-MoE behavior class already ships in competitor products (KTransformers, MoE-Infinity, Fiddler, llama.cpp `-ot`); the trunk-LR mechanism is published (SLCA).

## TL;DR

mini-AGI splits cleanly into *the thing that learns* (experts, routers, halting head — gradient territory) and *the thing that governs* (admission, growth, pruning, rate control, storage). The governing layer touches **zero gradients** — closed-form arithmetic over counters, ticks, and running sums. Our workspace already ships the storage half (freeze envelope, WAL commit batches, `kv_eviction`) and one eviction-scoring half; the **policy** halves are the gaps:

1. **`admission`** — hysteresis admission for a fixed resident set: candidate displaces the weakest *evictable* resident only when beating it by `margin`; newcomer un-evictable for `dwell` measured from its **admission tick** (residents are touched every cycle, so a last-use clock would make them permanently young — measured the hard way at `paged.py:242-247`); exactly one never-admitted item gets a fair turn per cycle (terminating sweep, re-arms on growth); apply by identity so loads = true set delta.
2. **`rate_control`** — a multiplicative rate controller on ANY noisy scalar evidence series: two exponentially-weighted least-squares fits as 7 running sums (slow = significance, fast = "is it still happening"), verdict = `min(t_slow, EFFECT·e_slow, EFFECT·e_fast)` where the deciding quantity is the **effect size** `e = slope/σ_resid` (the t-statistic measures watch-time, not progress), asymmetric tanh nudges (up = slow probe; down = responds to magnitude of deterioration), confirmed regime-jump step (×2 + fit reset). Kills the cosine-with-horizon assumption every riir-train trainer currently ships.
3. **`dying`** — dual-threshold staleness death metric: `d = min(now − last_addressed, own_age)/window`, newborns pinned 0, ONE definition read at two thresholds (0.75 = stop feeding/growing; 1.0 = delete). Anti-predictive-magnitude lesson: the busiest experts carry the smallest gates — magnitude is anti-predictive of being wanted again (`paged.py:292-299`).

## Mechanism inventory (condensed; full table in the panel record)

| # | mini-AGI mechanism | Evidence | Workspace analog | Verdict |
|---|---|---|---|---|
| 1 | Recency-normalized eviction scoring | `paged.py:899-1005` | `katgpt-core/src/kv_eviction/` (`usage_rate_eviction`, Plan 585 / Bench 697: `cum_mass/max(1,age)`) | **ships** (keep-side) |
| 2 | Atomic weights-as-files, identity-keyed | `store.py:27-29,144-147` | freeze envelope (BLAKE3, versioned), local_kv WAL/`CommitBatch` | **ships** |
| 3 | Aux-state travels with entity through swaps | `paged.py:27-33,717-762` | freeze/thaw whole-snapshot; uid-keyed Pods | **partial** |
| 4 | Best-state-advance-on-improvement dir + revert-on-divergence | `store.py:432-451` | `fix_verify` auto-revert (edit-level); kill-switch ladders | **partial** |
| 5 | Dying metric (age-clamped staleness, dual threshold) | `paged.py:284-325` | `decay_confidence` (perceptual fade, no deletion semantics) | **gap** → primitive C |
| 6 | Hysteresis admission (margin+dwell+since-clock) | `paged.py:569-627` | `SolverRouter.hysteresis_pct` (dispatch band, no resident set) | **gap** → primitive A |
| 7 | Identity-matched swap (loads = set delta) | `paged.py:690-714` | `PagedKVCache` free-list (layout only) | **gap** (as policy) |
| 8 | Terminating cold-start fair-turn sweep | `paged.py:620-626` | none | **gap** → primitive A |
| 9 | Exploration reserve (longest-unseen share) | `paged.py:629-657` | none | **gap** |
| 10 | Demand on context-bearing hidden states, not raw embeddings | `paged.py:479-566` | `hga_forward.rs:146-157` (attention-scored working set); `RerankMode::Structural` | **partial** |
| 11 | Dual-EWLS effect-size rate controller | `plasticity.py:20-33,63-358` | `BreakevenTracker` (level-EMAs, no slope/actuation) | **gap** → primitive B |
| 12 | GradSNR (2-scalar gradient noise scale) | `optim.py:8-52` | none | **gap** (training-side meter; rides Plan 416) |
| 13 | Speculative growth + survival trials + 5 brakes | `pool.py:555-732` | `frontier_miner` acceptance invariant; `can_freeze_covariability` | **partial** |
| 14 | Recombination birth (whole hidden units, ~16 parents) | `paged.py:766-842` | `archetype_blend_shard` (mixtures, not part-crossover) | **gap** (fusion idea, neuron-db) |
| 15 | Adaptive-depth halting at inference | `recur.py:285-318` | `GainCostLoopHalter` (cognition loops) | **partial** |
| 16 | Capacity-factor drop = router-collapse canary | `pool.py:477-499` | `prefill_gv_fallback_counts()` | **partial** |
| 17 | want_k/pressure EMAs (pre-plateau capacity signal) | `pool.py:368-379` | none | **gap** (minor) |

## Signal-diffs on the coverage dismissals (§3.6 discipline)

- **#10 "demand prediction" ≠ covered by `hga_forward`:** hga scores working-set fetch by attention over the *current* context; mini-AGI scores demand over the **whole pool** (router speaks only for trained/resident experts; per-expert SVD keys speak for all, gate-earned-scaled pre-normalization with `key_floor` + trial waiver). Delta = whole-pool describability offline + earned-discount placement. Partial, real gap on the pool side.
- **#5 "dying" ≠ covered by `kv_eviction::score`:** `cum_mass/age` is a keep-ranking **usage rate**; dying is its delete-side complement, an **unaddressed fraction** with age-clamping and dual thresholds. Different consumers (capacity pruning vs KV row selection).
- **#14 "recombination" ≠ covered by `archetype_blend_shard`:** blends are committed *mixtures* of whole latent directions; mini-AGI's birth is **part-crossover of functional units** ((w1[u], w3[u], w2[:,u]) columns across ~16 parents) with measured cosine-novelty vs parents (clone+noise ≈ 0.98 = not novel; random = useless). Fusion candidate, not a covered mechanism.
- **Validation rows (design-confirming, no file):** demand-on-hidden-states validates `RerankMode::Structural` default (context-bearing features beat raw embeddings — same lesson, both codebases); "writing costs more depth than reading" (9.9 vs 8.0 rows/char) parallels our prefill-vs-decode budget asymmetry.

## Prior art (§4 — searched, cited)

- **Trunk-LR split (training row):** **PUBLISHED** — SLCA (Zhang et al., CVPR 2023) / SLCA++ (TPAMI 2024): selectively slow backbone LR vs head; layer-wise LR decay is standard practice since BERT fine-tuning. ⇒ No novelty claim; it is a *recipe adoption* (the effect size on our MLA-MoE is still the measurement Plan 416 owns).
- **MoE expert paging/offload:** **ABUNDANTLY PUBLISHED** — Eliseev & Mazur 2023 (speculative prefetch + LRU cache), Pre-gated MoE (ISCA 2024), MoE-Infinity (EuroSys 2025), Fiddler (ICLR 2025), KTransformers (2025), llama.cpp `-ot`/`exps=CPU`. ⇒ The container is not ours to claim; only the **policy layer** (margin/dwell/since-clock/fair-turn) is candidate-novel.
- **Admission policy:** nearest published cousin is **TinyLFU/W-TinyLFU** (Caffeine; admission by frequency-sketch candidate-vs-victim compare) — *added from coordinator knowledge, not from the subagent sweep*; delta = dwell window + admission-tick clock + terminating fair-turn sweep + multiplicative margin (no sketch, works on arbitrary want-scores).
- **t-stat/effect-size LR controller:** no exact match found (ReduceLROnPlateau = patience threshold; Sakana 2025 = learned policy; Xu 2019/Wu 2018 = RL schedules). **Appears unpublished as stated.**
- **Growth-by-recombination + survival trials:** components published (DERN EMNLP 2025 recombination-for-pruning; GoD-MoE AAAI 2026 + LEGO ICML 2026 growth-on-demand; NEAT 2002 evolutionary); the specific growth-side combination appears unpublished.
- **Recency-as-capacity-pruning (not cache):** LRU expert-cache eviction is standard (FlashMoE 2026 benchmarks against it); staleness pruning of *model capacity* (experts/rules/shards) found no published match. **Appears unpublished as stated.**

## Novelty gate scoring (per primitive, §1.5)

| | `admission` | `rate_control` | `dying` |
|---|---|---|---|
| Q1 no prior art (exact form) | ~yes (TinyLFU delta above) | yes (exact form) | yes (capacity side; LRU cache side published) |
| Q2 new behavior class | no — paged-MoE products carry the class | partially (no runtime rate controller ships here) | no — pruning by staleness is a metric, not a class |
| Q3 product selling point | indirect (via serving on small GPUs) | no direct consumer yet (report-first law) | indirect |
| Q4 force multiplier | yes (kv_eviction + PagedKVCache + hga + zone_cache + thermal tiers) | yes (trainers + runtime knobs) | yes (compactor + healer retirement + belief GC) |
| **Tier** | **GOAT** (open primitive) | **GOAT** (open primitive) | **GOAT** (open primitive) |

Not Super-GOAT: no all-4 column. All three file under Issue 873 (closed 2026-09-25, HISTORY.md § Issue 873) as opt-in katgpt-core features, GOAT-gated, report-first where the R135/Bench-047 law applies.

## Fusion (the Super-GOAT attempt, recorded honestly)

**Governed paged pool** = `admission` × `dying` × `rate_control` × shipped substrate (`kv_eviction` + `PagedKVCache` + thermal tiers + `SolverRouter`): a resident-set pool whose admission is hysteretic, whose deletion is staleness-based with one definition at two thresholds, and whose exploration/consolidation rates adapt by measured evidence. Consumers: riir-ai MoE serving working sets (hga), KV page pools, riir-neuron-db `zone_cache`, riir-clippy corpus governance. Failed the Super-GOAT gate only on Q2/Q3 — but it is the wiring story that makes the three primitives more than isolated additions, and it is what Issue 873's tasks build toward.

## Routing (per-track, no cascade)

- **katgpt-rs:** this note + Issue 873 (primitives A/B/C). `rate_control` is the modelless half of a dual-track pair (below).
- **riir-train:** Research 456 (recipe table) + Plan 416 (trunk-LR split, plasticity consumption, forgetting probe, riders). Filed per Path 0.5; QUEUED behind the perf league per owner priorities — filing ≠ scheduling.
- **riir-neuron-db / riir-clippy:** consumers of `dying` (shard/rule retirement) and fusion idea #14 (recombination birth for consolidation/corpus growth) — recorded here, no files (consumer-side wiring is theirs to file when scheduled).

## Weakest points (named by the author)

1. The "appears unpublished" verdicts for `rate_control`/`dying`/recombination rest on a search pass with intermittent backend failures (3–4 phrasings each) — a Scholar citation-graph walk of DERN/LEGO/GoD-MoE full texts could still surface a match. The TinyLFU caveat on `admission` was added by the coordinator, not the sweep.
2. The mechanism table came from a subagent sweep; I spot-verified `kv_eviction`, `SolverRouter.hysteresis_pct`, `katgpt_transformer::moe` (Kimi-K3), and `elasticity_gated_update`/`qsg` (our "plasticity" = per-NPC drift α, a different concept) — but `EigenbasisTracker`'s source path and `hga_forward.rs` line cites are unverified by me.
3. The GPU-hour figures in Plan 416 are paper-derived (mini-AGI's 8 GB-3070 reference scaled to 4090), not measured.

## Verdict record (§5)

Claude verdict ping-pong ran 2026-09-22 via the **fallback sub-agent path** (equal verdict per skill §5): the `claude_code` reviewer backend timed out twice at 180s; the same Summary + `#Verdict:` contract went to a generic sub-agent reviewer, which read all four filed files and replied **#Verdict: AGREE** with one non-blocking citation correction (applied: Research 456 row 2's second cosine cite replaced with the verified entities). No verdict-bearing section changed post-AGREE.
