# Issue 931: symmetric_eig QL diverges on graded matrices — loud NaN panic + a silent-overflow class under scale normalization

**Status:** Open — P1 (substrate robustness; blocks the riir-ai Issue-1056 style-family audit arm). Found by the ai-1056 basis_purity_audit run (Bench 980) against katgpt `015203f9f`.

## Problem

`crates/katgpt-core/src/linalg/symmetric_eig/mod.rs` (the NR-style QL with
implicit shifts) fails on GRADED symmetric matrices — large diagonal spread
with mixed-magnitude off-diagonals. Two measured failure modes on the SAME
input class (the riir-ai style-shard covariance; d=64, built from 1024
real-pipeline snapshots):

1. **Loud NaN panic** (the raw input):
   `symmetric_eig: QL failed to converge at l=0 after 200 iterations
   (d[l..l2]=[NaN, NaN], e[l..l2]=[NaN, NaN])` — raised inside
   `basis_split::split` → `build_zca_whitener` → `symmetric_eig(sigma)`.
2. **Silent overflow** (after a mathematically-invariant uniform scale of
   sigma+moment to O(1) diagonals): the solver returns "success" with
   eigenvalues at ±1e78 (spectrum printed in the bench record) — a
   **confidently wrong** result, strictly worse than the panic.

Measured properties of the triggering matrix (numpy cross-check on the
dumped bytes at the time of the run; the dump is reproduced by the gated
test):

- symmetric exactly (sym err 0.0), all entries finite;
- eigenvalue condition ≈ 7.3e5 (min eig 7.08e-8 > 0, max 5.14e-2) — benign
  by LAPACK standards;
- `numpy.linalg.eigvalsh` + `eigh` solve it to full precision — the input
  is well-posed; the QL port is what fails;
- a +1e-3·max-diag ridge (lifting min eig to ~8e-6, cond ~6.5e3) does NOT
  repair it — the panic persists, so the fragility is not simply absolute
  scale.

## Suspected root cause

The NR C recipe's negligible-off-diagonal test
`(|e[m]| + dd) == dd || |e[m]| <= eps·global_max_d` — on a graded matrix a
tiny `e[m]` can sit ABOVE the rounding threshold of an equally-tiny `dd`
while BELOW the global-eps check's resolution relative to its local block,
so the deflation never fires; the shift computation then under/overflows
(depending on absolute scale — hence the two failure modes). Classic
graded-matrix QL pitfall; LAPACK's `dstev`-class codes use per-block local
scaling.

## Reproducer

Run the riir-ai gated test (opt-in feature `basis_purity_audit`, Bench 980)
— it computes sigma+moment through the real `PlayerStyleCapture` pipeline
(deterministic, seeded), dumps both matrices to
`E:/git/_sync/family_diag.txt` BEFORE the solve, and the style arm
loud-blocks on this issue. Feeding the dumped SIGMA to `symmetric_eig`
directly reproduces the panic.

## Fix direction (not this issue's scope)

- Per-block local scaling before the shift (the LAPACK approach), or
- Wilkinson-shift convergence guard + fallback: on non-convergence at
  local scale, rescale the active block and continue, and
- A loud guard that REFUSES (Err) on non-finite intermediates instead of
  returning overflowed "eigenvalues" — the silent mode is the unacceptable
  one.

## Tasks

- [ ] T1 Standalone failing test in katgpt-core (`symmetric_eig` on the
      dumped/reconstructed graded matrix) — no ai deps.
- [ ] T2 Repair (local-block scaling or guarded fallback) + the
      non-finite-intermediate loud refusal.
- [ ] T3 Re-run the ai-1056 style arm (its S1 numbers unblock).

## Acceptance

`symmetric_eig` returns finite, accurate eigenvalues on the reproducer
(±1e-9 vs LAPACK), never panics-with-NaN and never returns non-finite
"results"; the ai-1056 style arm unblocks.
