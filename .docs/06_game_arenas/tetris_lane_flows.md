# Tetris arena lanes — how each lane picks a spot (the site figures)

> **Purpose:** the source of truth for the "How each lane decides" figures on
> [reflex.gist.rs/arena](https://reflex.gist.rs/arena/). Each `mermaid` block
> below renders ONE SVG (named by its `%% file:` line); the renderer is
> `reflex-site/scripts/render_tetris_flows.py`, which writes the SVG beside
> this doc AND into `reflex-site/assets/` (the mirror law of riir-reflex
> `.docs/03_decision_flow` — re-render from here, never hand-edit an SVG).
> Labels stay short: the figures must read at page scale; the numbers live
> in the records cited under each block.

Every lane plays the same game: a seeded 7-bag, a real hard drop from the
top (`laya-tetris-v3`), guideline scoring. They differ only in HOW a
landing spot is chosen.

## 1 · laya (Rust) and laya (Python) — the model reads every spot

Both lanes run the same open weights; Rust is the served port, Python the
torch reference it is measured against.

```mermaid
%% file: tetris_flow_laya.svg
%% aria: laya lanes: code lists every landing spot, writes each as an English sentence, the laya model answers Does the stack look clean for each, and the piece goes to the highest probability
flowchart TB
    subgraph R1[" "]
        direction LR
        B["board +<br/>falling piece"] --> L["list every<br/>landing spot<br/>~17 per piece"] --> S["write each spot<br/>as a sentence<br/>'no holes, sits flat…'"]
    end
    subgraph R2[" "]
        direction LR
        M["laya model reads it<br/>'Does the stack<br/>look clean?'"] --> P["P(clean)<br/>per spot"] --> A["play the<br/>highest"]
    end
    S --> M
```

Cost: one model forward per spot (~0.1–0.7 s each on an M3); it sees one
piece — no preview, no plan. Records: katgpt-rs `.benchmarks/892_laya_h2h.md`.

## 2 · KatGPT modelless — laya's answers, distilled into a µs head

```mermaid
%% file: tetris_flow_modelless.svg
%% aria: KatGPT modelless lane: the same spot sentence is decoded back into board features, scored by a small linear head fitted on laya's recorded answers, and the highest score is played, in microseconds inside WebAssembly
flowchart LR
    S["spot sentence<br/>(same as laya's)"] --> D["decode to features<br/>holes · flat · height<br/>side · clears"]
    D --> H["linear head<br/>fitted on laya's<br/>recorded answers"]
    H --> A["play the<br/>highest"]
    W["runs in your tab<br/>WebAssembly · ~1 µs"] -.-> H
```

It imitates laya (44/120 agreement on the pinned corpus), so it plays like
laya — only ~10⁵× faster. Records: riir-reflex game heads, Plan 607.

## 3 · raw baseline — the engine with its heads removed

```mermaid
%% file: tetris_flow_raw.svg
%% aria: raw baseline lane: the corpus engine without fitted heads finds the spot sentence off-corpus, abstains, and the abstain plays as a labelled random spot, the honest floor
flowchart LR
    S["spot sentence"] --> E["corpus engine<br/>no fitted heads"]
    E --> Q{"on-corpus?"}
    Q -->|"no"| R["abstain →<br/>random spot<br/>(labelled)"]
    Q -->|"yes"| A["play the<br/>highest"]
```

The honest floor every other lane is measured against.

## 4 · KatGPT rulebook search — plan three pieces ahead

The Issue 892 hybrid champion (genome `68cae9d382014662`): no model, no
sentence — it searches placements and scores the boards with a strategy
rulebook.

```mermaid
%% file: tetris_flow_rulebook.svg
%% aria: KatGPT rulebook search: try every spot for the falling piece, every spot for the preview piece keeping the best six, then every piece still left in the bag; score each end board with the strategy rulebook, average over the unknown piece, take the best plan, and play its first move, in about six to nine milliseconds
flowchart TB
    subgraph R1["look ahead"]
        direction LR
        A["falling piece<br/>try every spot"] --> B["preview piece<br/>every spot<br/>keep best 6"] --> C["next piece = unknown<br/>try every piece<br/>left in the 7-bag"]
    end
    subgraph R2["decide"]
        direction LR
        V["score each end board<br/>with the rulebook"] --> G["average over the<br/>unknown piece ·<br/>best over the plan"] --> P["play the best<br/>first move<br/>~6–9 ms"]
    end
    C --> V
```

"Average over the unknown piece" is the point: a plan that only works if
the I-piece comes scores low, because it usually does not come.

## 5 · the rulebook's modes — play for tetrises only while it is safe

```mermaid
%% file: tetris_flow_modes.svg
%% aria: the rulebook's play modes: on a clean low stack it builds a nine-one stack and waits for tetrises; three or more covered holes switch it to downstack, a tall stack to survive, both using safe survival weights; a clean board returns it to build
flowchart LR
    BU["BUILD<br/>9-1 stack · keep the<br/>right edge open ·<br/>wait for tetrises"]
    DS["DOWNSTACK<br/>clear the lines<br/>over the hole"]
    SV["SURVIVE<br/>take any line<br/>stay low"]
    BU -->|"3+ holes covered"| DS
    BU -->|"stack ≥ 12 high"| SV
    DS -->|"holes dug out"| BU
    SV -->|"back under 12"| BU
```

Build uses the self-evolved score weights (the 9-1 stack emerged from the
climb — nobody hand-coded it); Downstack and Survive use the proven
survival weights. The mode is re-read from the board before every piece
(Survive wins when both fire). Records: katgpt-rs `.benchmarks/892_tetris_rulebook_arena.md`.
