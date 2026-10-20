# KronCell — the pair-addressed scalar-target delta-rule cell

> **Status: OPT-IN, G0-FALSIFIED for promotion (Bench 951, 2026-10-20).**
> `kron_memory = []` / `kron_chunk_wy = ["kron_memory"]` — diagnostic-grade,
> sound machinery (all 19 G1 property tests pass), but the headline
> surgical-unlearning claim did not survive its own falsifier: at matched
> state bytes and low load the plain one-hot slot table wins both G0
> statistics. The interference-envelope sizer (G5a) is the deliverable that
> DID validate. Source: `crates/katgpt-core/src/kron_memory/` (Issue 954,
> Research 630, arXiv:2610.26834).

## The cell

A new target contract on the associative-memory ladder — the rungs table is
Research 630 §2.1. Where the vector-target delta rule (`delta_mem`) regresses
a VECTOR `v` on key `k`, KronCell regresses a **SCALAR** `c` on the rank-1
feature `a = k ⊗ v`:

```text
state     S ∈ R^{DK×DV}          (fixed Pod: DK·DV·4 B — 16 KB @ 64×64)
read      r(k, v) = Σᵢ k[i]·(Σⱼ S[i][j]·v[j])     [pinned order, see below]
write     S ← S − β·r̃·(v kᵀ),   r̃ = r − c        [one rank-1 axpy]
Newton    β* = 1/(‖k‖²‖v‖²)     — r̃ → 0 in one step (exact in ℝ)
governor  0 < β·‖k‖²‖v‖² < 2    — else the write is refused (divergence edge)
elision   r̃ == 0.0 exactly ⇒ the write is a bit-level no-op
```

Everything is allocation-free in the hot path; every dot goes through
`dot_f32_ordered` and the row fold is a fixed sequential `i`-ascending
accumulation (`KRON_READ_ORDER v1` — the Issue 940 determinism law). The
exact-zero elision is therefore a TRUE no-op: the frozen-state identity
fixed point is hash-pinned over 64 replayed writes (G1).

## Size disclosure (never "16 KB" alone)

| component | bytes @ 64×64, N=64 |
|---|---|
| state Pod | 16,384 |
| pair registry `N·(DK+DV)·4` | 32,768 |
| undo record `(DK+DV)·4 + 4` | 516 |
| health tracker (opt-in) `(DK²+DV²+2·DK·DV)·4` | 32,784 |

Helpers: `kron_state_bytes`, `kron_registry_bytes`, `kron_health_bytes`.

## The interference kernel and audited unlearning

Interference on a stored pair `(kᵢ, vᵢ)` from a write `(k, v, β, c)` is
EXACTLY `Δᵢ = −β·r̃·(vᵢᵀv)·(kᵀkᵢ)` — the product of value overlap and key
overlap. Either factor ≈ 0 protects the pair (the double gate), and the
damage is computable BEFORE the write: `collateral_audit` enumerates the
cell's pair registry into caller scratch (`O(N·(d_k+d_v))`, zero alloc; the
returned count-vs-registry-len pair makes truncation loud).

`unlearn_audited(k, v, τ, policy)` is the surgical erase: a `c = 0` Newton
write that audits itself first. Under `τ` it applies; over `τ`,
`CollateralPolicy::Refuse` leaves the state untouched and `Repair` writes
back every collided pair's old readout with one closed-form Newton write
each — the repair's measured residual is the audit kernel's empirical
validation (it lands in `CollateralReport::repair_residual_max`).

**What G0 measured (Bench 951):** against the vector-target Newton erase,
the double gate IS 17× more surgical on distributed keys (mean collateral
0.00482 vs 0.08715, paired LB95 +0.081 > 0) — but against a plain one-hot
slot table at 1.2% load the table wins both statistics (collateral 0.00016,
recall 0.994 vs 0.780). The falsifier fired; the flag stays opt-in forever.
Re-open bar: a G0′ at matched load ≥ 50% with its own pre-registration.

## Undo

`undo_last()` removes the MOST RECENT applied write (single-level — each
write replaces the record; arbitrary-suffix undo is replay-based, per the
issue's disclosure) to within `EPS_UNDO = 1e-5` per element (two f32
roundings; the fp round-trip is NOT bit-exact). For bit-exact undo,
`write_snap` captures the touched rows into the caller's `TouchedSnap<DV, CAP>`
(`CAP = DK` always sufficient; sparse designed addresses can run `CAP ≪ DK`;
insufficient capacity refuses loudly, never truncates) and `undo_snap`
restores them bit-for-bit. The chunk lane's composite apply is not
rank-1-invertible — after `write_chunk_wy`, `undo_last` reports false until
the next single write.

## Health gauges — lower-bound heuristics, signals never guarantees

Opt-in via `with_health(alpha)` (roughly doubles the write cost — the S-sized
cross moment dominates; do not tax cells nobody reads gauges from).
`health(beta, theta)` returns:

- `lambda_min_k/v/a` — λ̂_min of the EMA marginal Grams and their Kronecker
  product, tolerance-thresholded on the OBSERVED subspace (1e-6·λ_max floor):
  rank-deficient streams keep the gauges finite instead of sending the
  half-life to ∞.
- `half_life_writes = ln2/(β·λ̂_min(Ā))` — first-order writes-to-halve for a
  steady-state perturbation. `None` when λ̂ is unavailable — finite by
  absence, never ∞.
- `norm_budget_exceeded` — `‖S‖²_F > θ·4·Ê[c²]/λ̂_min(Ā)`. The envelope reads
  the EMA Gram's WINDOW while the no-decay state integrates the whole
  history: it fires when the accumulation crosses ≈ 4/λ̂ (the
  stationary-regime exit), pinned deterministically in G1 (α=1, orthogonal
  addresses: quiet at P=3, fired at P=5).
- `independence_violation = ‖EMA(k⊗v)‖_F/√(tr K̄ · tr V̄)` — the MEASURED
  dependence between marginals (0.73 on a v=k stream in G1): how much the
  factorized envelope should be distrusted on THIS stream.

## Consolidation

`consolidate_ls(ridge)` refits `S := K̄⁻¹ B V̄⁻¹` over the tracked moments
(the LS cross-moment `B = EMA(c·k⊗v)` rides the tracker, one rank-1 EMA per
write) — resetting the accumulated O(β) jitter. Eigensolves go through the
shipped `linalg::symmetric_eig` (f64) with Tikhonov ridge on the spectrum
(rank-deficient streams need `ridge > 0`); a non-finite refit refuses
`Singular` with the state untouched. The solve is exact LS regardless of
independence; the RETENTION envelope stays independence-conditional.
Invalidates the undo record and clears the registry (addresses survive as
vocabulary; readouts are refit).

## The chunk lane — opt-in, UNPROMOTED

`kron_chunk_wy` derives the residual-coupling system `(I+N)·r = r⁰` with
`N[l][m] = β_m·(G_kk ⊙ G_vv)[l][m]` (the Hadamard product of the Gram
triangles, COLUMN-scaled by the earlier write's β) — one unit-LT solve, one
fused rank-1 sum, features never formed. Equality vs sequential is the
pinned 1e-3 TOLERANCE (rounding differs by construction); sequential stays
canonical for anything synced or replayed. **Bench 951 G2: 0.56× — SLOWER
than sequential** (the Grams + fused apply are ~1.5× the MACs and the
sequential row axpys already vectorize well). The lane stays opt-in,
unpromoted; re-open bar: blocked-GEMM Grams via the `kron_tile`/
`simd_matmul` family — priced, not attempted.

## The validated deliverable — the interference-envelope sizer

The self-consistent first-order retention law for Newton writes on
independent normalized addresses:

```text
MSE(P) ≈ σ²_c / (1 − x),   x = P·β²/(d_k·d_v),   valid while x < 1
```

(Each subsequent write adds `E[r²]/(d_k·d_v)` to a stored pair's error;
`E[r²] ≈ σ²_c + MSE` closes the loop.) Measured on the Bench 951 erosion
run: ratio 0.83 and 0.67 at x = 0.0625 and 0.25 — inside the [0.5, 2]
band; the x → 1 point is the model's declared limit (the measured curve's
E[r²] shrinkage regularizes past it). **Promoted for NEW-config sizing in
the sub-saturation regime.** The measured mechanism beside it: a no-decay
scalar cell's erosion SATURATES at ≈ σ²_c — a stored pair's read
decorrelates from its target as the state superposes; the capacity of this
contract on random addresses is an interference floor, not a retention
cliff.

## Uncertainty posture

The analytic CI around `read` was DROPPED at G5b (Report the Floor): the
residual-signal candidate beat the canonical conformal-naive floor on
coverage (0.896 vs 0.203) and Winkler (2.62 vs 11.64) but lost CRPS
(2.156 vs 2.036) — the rule requires all three, so the cell ships
SIGNAL-ONLY: `residual_signal()` is the EMA of |residual|, a scale signal
for callers, never an interval. (The floor's 0.203 coverage on
fresh-address associative streams is itself recorded: seasonal-naive is the
wrong point forecaster there.)

## See also

- [`lattice_memory.md`](lattice_memory.md) — the vector-target E2LSH lattice (the host candidate for a lattice of Kron cells)
- [`../05_training/`](../05_training/) — the trained-gate track is riir-train Plan 468 (fixed schedules only here)
- Research 630 (the distillation) · Bench 951 (the GOAT record) · Issue 940 (the determinism law this cell implements)
