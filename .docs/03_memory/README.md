# Memory — Modelless Memory Primitives

> **What we sell here.** The memory substrate that lets a frozen model remember
> across context windows without gradient updates: O(1) routing slots,
> O(√N) factored retrieval, hash-addressed conditional patterns, attractor
> belief kernels, multi-modal sense composition, and eviction-time consolidation.

## Fusion map — the memory stack

```
   sense_composition.md (multi-modal sense → latent input)
        │
        ▼
   raven_rsm.md (O(1) routing slot memory)
        │
        ├── product_key_memory.md (O(√N) factored retrieval)
        ├── engram.md (hash-addressed conditional pattern memory)
        ├── lattice_memory.md (E2LSH delta-rule cell lattice — the candidate fifth class, envelope-bounded)
        ├── kron_memory.md (pair-addressed SCALAR-target delta cell — opt-in, G0-falsified for promotion)
        ├── micro_belief.md (attractor / leaky belief kernel)
        ├── drift_segment.md (drift-segmented multi-state memory, katgpt-kv)
        └── chunked_content_store.md (content-addressed Merkle blob store)
              │
              ▼
        sleep_consolidation.md (bake cache → fast weights at eviction)
```

| Doc | Role |
|---|---|
| [`raven_rsm.md`](raven_rsm.md) | Raven RSM — O(1) routing slot memory |
| [`product_key_memory.md`](product_key_memory.md) | PKM — O(√N) factored retrieval memory (Plan 408, default-ON) |
| [`engram.md`](engram.md) | Engram — hash-addressed conditional pattern memory (Plan 299) |
| [`lattice_memory.md`](lattice_memory.md) | LatticeMemory — E2LSH delta-rule cell lattice, the candidate fifth retrieval class (Plan 619, opt-in — NOT promoted at 10⁶; the support×capacity law is the deliverable) |
| [`kron_memory.md`](kron_memory.md) | KronCell — pair-addressed scalar-target delta-rule cell (Issue 954, opt-in — G0 falsifier FIRED at Bench 951: the slot table wins at low load; the interference-envelope sizer is the deliverable) |
| [`micro_belief.md`](micro_belief.md) | MicroRecurrentBeliefState — attractor + leaky belief kernel (Plan 276) |
| [`sense_composition.md`](sense_composition.md) | NPC sense composition (Plans 221/230/235/236/237) |
| [`drift_segment.md`](drift_segment.md) | DriftSegmentStore — training-free drift-segmented multi-state memory (Issue 652 / Research 482, katgpt-kv, opt-in — Bench 635 GOAT PASS) |
| [`chunked_content_store.md`](chunked_content_store.md) | ChunkedContentStore — Lore-distilled content-addressed Merkle blob store (Plan 448, default-ON since Phase 19b promotion fix-up 2026-07-18) |
| [`sleep_consolidation.md`](sleep_consolidation.md) | Sleep consolidation — offline recursive memory consolidation at eviction (Plan 154, default-ON) |

## See also

- [`../04_calibration/faithfulness_probe.md`](../04_calibration/faithfulness_probe.md) — causal diagnostic for *injected* memory
- [`../01_orientation/architecture.md`](../01_orientation/architecture.md) § Sleep-Time Query Anticipator — the artifact-emission sibling of sleep consolidation
