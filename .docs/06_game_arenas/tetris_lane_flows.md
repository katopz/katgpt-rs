# Tetris arena lanes — how each lane picks a spot (the site figures)

> **Purpose:** the source of truth for the "How each lane decides" figures on
> [reflex.riir.ai/arena](https://reflex.riir.ai/arena/). Each `gfflow` block
> below renders ONE figure (named by its `file` key) as the family swimlane +
> the 390 px card list + the step-through walk; the renderer is
> `reflex-site/scripts/render_flows.py`, which writes the outputs beside this
> doc AND into `reflex-site/assets/` (the mirror law — re-render from here,
> never hand-edit an SVG). Labels stay short: the figures must read at page
> scale; the numbers live in the records cited under each block.
>
> The walks' IN/OUT payloads are derived by
> `reflex-site/scripts/build_arena_flow_walks.py` from the arena's RECORDED
> games (`reflex-site/arena/demo_oracle.json`) — captured, never typed (the
> family numbers law applied to payloads).

Every lane plays the same game: a seeded 7-bag, a real hard drop from the
top (`laya-tetris-v3`), guideline scoring. They differ only in HOW a
landing spot is chosen.

**Numbering note (Plan 620, the branch rule — design guide §8.2).** Numbers
follow the PATH. The raw lane is the one flow with a fork: the engine's
verdict branches, so its two alternatives share 3 — **3a** *in corpus* →
play the highest, **3b** *off corpus* → abstain, and the abstain plays a
labelled random spot. The rulebook's last step loops back to 1 (the search
restarts every piece — a back edge, never a number); the modes figure is a
cycle of back edges: BUILD branches to **2a** DOWNSTACK (3+ covered holes)
and **2b** SURVIVE (a 12+ row stack), and both hand back to BUILD. Every
step is `live`: all five lanes are real, recorded and measured.

## 1 · laya (Rust) and laya (Python) — the model reads every spot

Both lanes run the same open weights; Rust is the served port, Python the
torch reference it is measured against. The figure's lanes are the
pipeline's phases: the game deals a position, the harness enumerates the
spots and writes the sentence, the model answers.

```gfflow
file  = "tetris_flow_laya.svg"
title = "laya (Rust + Python): the model reads every landing spot"
accent = "reflex"
intro = "Press play to walk one recorded turn: the opening position of the seed-607 game, the same one every lane plays."

[[lane]]
id = "game";    label = "The game";   note = "seeded 7-bag · hard drop"; color = "reflex"
[[lane]]
id = "code";    label = "The harness"; note = "enumerates + writes";      color = "reflex"
[[lane]]
id = "model";   label = "The model";  note = "open weights · Rust or Python"; color = "reflex"

[[step]]
id = "turn"; n = "1"; lane = "game"; col = 0
title = "The turn arrives"; body = "board plus falling piece from the seeded 7-bag"
status = "live"
[[step]]
id = "list"; n = "2"; lane = "code"; col = 1
title = "List every spot"; body = "each rotation in each column — the futures to judge"
status = "live"
[[step]]
id = "write"; n = "3"; lane = "code"; col = 2
title = "Say it in words"; body = "holes, surface, side, height — one sentence per spot"
status = "live"
[[step]]
id = "read"; n = "4"; lane = "model"; col = 2
title = "The model reads"; body = "a laya forward pass per sentence"
status = "live"
[[step]]
id = "pclean"; n = "5"; lane = "model"; col = 3
title = "P(clean) per spot"; body = "a cleanliness probability for every landing"
status = "live"
[[step]]
id = "play"; n = "6"; lane = "game"; col = 3
title = "Play the highest"; body = "the top spot drops; the next turn begins"
status = "live"

[[edge]]
from = "turn"; to = "list"
[[edge]]
from = "list"; to = "write"
[[edge]]
from = "write"; to = "read"
[[edge]]
from = "read"; to = "pclean"
[[edge]]
from = "pclean"; to = "play"

[[walk]]
title = "The turn arrives"
text  = "One turn of the recorded seed-607 game: the board, the falling piece, and the nine landing spots the harness will judge. Every lane in the arena plays this same stream — what differs is only how the pick is made."
steps = ["turn"]
in    = { lang = "json", src = "data/flows/tetris_laya/01_in.json", label = "the recorded opening turn" }
[[walk]]
title = "List every spot"
text  = "The harness enumerates every legal landing: each rotation of the piece in each column, hard-dropped. For the opening O piece that is nine distinct futures — the count rides the recorded turn above."
steps = ["list"]
[[walk]]
title = "Say it in words"
text  = "Each spot becomes one English sentence — the holes it leaves, how it meets the surface, where it lands, how tall the stack stands. The words are pinned by the grammar; every lane reads the same ones."
steps = ["write"]
[[walk]]
title = "The model reads"
text  = "The laya model runs one forward pass per sentence. Rust is the served port; Python is the torch reference it is measured against — the same open weights either way."
steps = ["read"]
[[walk]]
title = "P(clean) per spot"
text  = "Every landing spot comes back with a cleanliness probability. The recorded opening turn's nine answers are below — watch how flat they are on an empty board: the model is nearly indifferent, and small differences decide the pick."
steps = ["pclean"]
out   = { lang = "json", src = "data/flows/tetris_laya/01_out.json", label = "P(clean) per spot · the pick" }
[[walk]]
title = "Play the highest"
text  = "The highest-probability spot is played and the piece drops; the next turn starts the loop again. One model forward per spot is the whole cost of this lane."
steps = ["play"]
out   = { lang = "json", src = "data/flows/tetris_laya/01_out.json", label = "the played spot" }
```

Cost: one model forward per spot (~0.1–0.7 s each on an M3); it sees one
piece — no preview, no plan. Records: katgpt-rs `.benchmarks/892_laya_h2h.md`.

## 2 · KatGPT modelless — laya's answers, distilled into a µs head

```gfflow
file  = "tetris_flow_modelless.svg"
title = "Reflex · modelless: laya's answers distilled into a µs head"
accent = "reflex"
intro = "Press play to walk the same recorded turn through the distillation: the same sentence in, a different pick out — in microseconds."

[[lane]]
id = "wire";   label = "The sentence"; note = "the same wire laya reads"; color = "reflex"
[[lane]]
id = "decode"; label = "The decode";   note = "";                          color = "reflex"
[[lane]]
id = "head";   label = "The head";     note = "WebAssembly · your tab";    color = "reflex"

[[step]]
id = "sentence"; n = "1"; lane = "wire"; col = 0
title = "The same sentence"; body = "the spot sentence laya reads, unchanged"
status = "live"
[[step]]
id = "decode"; n = "2"; lane = "decode"; col = 1
title = "Decode to features"; body = "holes · flat · height · side · clears, as numbers"
status = "live"
[[step]]
id = "head"; n = "3"; lane = "head"; col = 2
title = "A tiny linear head"; body = "fitted on laya's recorded answers — no model runs"
status = "live"
[[step]]
id = "play"; n = "4"; lane = "head"; col = 3
title = "Play the highest"; body = "microseconds in your tab; the top spot drops"
status = "live"

[[edge]]
from = "sentence"; to = "decode"
[[edge]]
from = "decode"; to = "head"
[[edge]]
from = "head"; to = "play"

[[walk]]
title = "The same sentence"
text  = "The distillation changes nothing about the input: the very same recorded opening turn the laya lane reads. That is the point — same question, a different (much cheaper) answerer."
steps = ["sentence"]
in    = { lang = "json", src = "data/flows/tetris_modelless/01_in.json", label = "the same recorded opening turn" }
[[walk]]
title = "Decode to features"
text  = "The sentence is decoded back into the board features it came from: holes left, how flat the surface sits, stack height, which side it lands on, lines cleared. A handful of numbers — no words reach the head."
steps = ["decode"]
[[walk]]
title = "A tiny linear head"
text  = "A small linear head, fitted offline on laya's recorded answers, scores the features. It imitates laya — 44 of 120 picks agree on the pinned corpus — so it plays like laya, only about a hundred-thousand times faster."
steps = ["head"]
out   = { lang = "json", src = "data/flows/tetris_modelless/01_out.json", label = "the head's answer to the same turn" }
[[walk]]
title = "Play the highest"
text  = "The top-scoring spot drops. The head runs as WebAssembly in your tab — it is the lane playing live in the arena below, one pick per piece at microsecond cost."
steps = ["play"]
out   = { lang = "json", src = "data/flows/tetris_modelless/01_out.json", label = "the played spot" }
```

It imitates laya (44/120 agreement on the pinned corpus), so it plays like
laya — only ~10⁵× faster. Records: riir-reflex game heads, Plan 607.

## 3 · raw baseline — the engine with its heads removed

The one flow with a **branch**: the engine's verdict forks at step 3.

```gfflow
file  = "tetris_flow_raw.svg"
title = "raw baseline: the engine with its heads removed — the honest floor"
accent = "reflex"
intro = "Press play to walk the same recorded turn through the bare engine: it abstains, and the abstain plays a labelled random spot."

[[lane]]
id = "ask";     label = "The sentence"; note = "unchanged";             color = "reflex"
[[lane]]
id = "engine";  label = "The engine";   note = "no fitted heads";       color = "reflex"
[[lane]]
id = "floor";   label = "The floor";    note = "the honest floor";      color = "reflex"

[[step]]
id = "sentence"; n = "1"; lane = "ask"; col = 0
title = "The same sentence"; body = "the spot sentence, sent to the bare engine"
status = "live"
[[step]]
id = "engine"; n = "2"; lane = "engine"; col = 1
title = "Corpus engine"; body = "no fitted heads — nearest corpus neighbours only"
status = "live"
[[step]]
id = "play"; n = "3a"; lane = "engine"; col = 2
title = "In corpus → play"; body = "a known shape: the highest neighbour wins"
status = "live"
[[step]]
id = "abstain"; n = "3b"; lane = "floor"; col = 2
title = "Abstain → random"; body = "no close neighbour: a labelled random spot plays"
status = "live"

[[edge]]
from = "sentence"; to = "engine"
[[edge]]
from = "engine"; to = "play"
[[edge]]
from = "engine"; to = "abstain"

[[walk]]
title = "The same sentence"
text  = "The same recorded opening turn, sent to the engine with every fitted head removed. Only the corpus remains: nearest-neighbour lookup over what it has literally seen before."
steps = ["sentence"]
in    = { lang = "json", src = "data/flows/tetris_raw/01_in.json", label = "the same recorded opening turn" }
[[walk]]
title = "Corpus engine"
text  = "With no heads, the engine can only compare the sentence against its corpus. The spot sentences of a real game are almost all new — so this branch is the one the recorded raw game took, five hundred and fifteen times in a row."
steps = ["engine"]
[[walk]]
title = "In corpus → play the highest"
text  = "When a sentence lands close to something the corpus holds, the engine answers like laya would and the highest-probability neighbour wins. In the whole recorded raw game this path never fired — that honesty is what makes it the floor."
steps = ["play"]
[[walk]]
title = "Off corpus → abstain, play random"
text  = "Every outcome comes back null — the engine abstains rather than guess. The abstain plays a random legal spot from a separate seeded stream, labelled as random in the replay so nothing pretends to be a decision."
steps = ["abstain"]
out   = { lang = "json", src = "data/flows/tetris_raw/01_out.json", label = "every P(clean) null · a recorded random pick" }
```

The honest floor every other lane is measured against.

## 4 · Reflexer (the rulebook search) — plan three pieces ahead

The Issue 892 hybrid champion (genome `68cae9d382014662`): no model, no
sentence — it searches placements and scores the boards with a strategy
rulebook.

```gfflow
file  = "tetris_flow_rulebook.svg"
title = "Reflex · rulebook: plan three pieces ahead, play one move"
accent = "reflex"
intro = "Press play to walk the search over one recorded position — the real seed-607 turn that ends in the recorded tetris. The board beside each step replays that exact position."

[[lane]]
id = "look";   label = "Look ahead"; note = "three pieces deep";     color = "reflex"
[[lane]]
id = "decide"; label = "Decide";    note = "score · average · pick"; color = "reflex"

[[step]]
id = "fall"; n = "1"; lane = "look"; col = 0
title = "Piece in play"; body = "every legal landing spot, listed"
status = "live"
[[step]]
id = "preview"; n = "2"; lane = "look"; col = 1
title = "Preview chained"; body = "every spot for the next piece, best six kept"
status = "live"
[[step]]
id = "bag"; n = "3"; lane = "look"; col = 2
title = "Third is unknown"; body = "branch over every piece left in the bag"
status = "live"
[[step]]
id = "score"; n = "4"; lane = "decide"; col = 2
title = "Score end boards"; body = "the strategy rulebook judges each one"
status = "live"
[[step]]
id = "average"; n = "5"; lane = "decide"; col = 1
title = "Average, then pick"; body = "over the unknown piece; sturdiest plan wins"
status = "live"
[[step]]
id = "play"; n = "6"; lane = "decide"; col = 0
title = "Play move one"; body = "the first move lands; the search restarts"
status = "live"

[[edge]]
from = "fall"; to = "preview"
[[edge]]
from = "preview"; to = "bag"
[[edge]]
from = "bag"; to = "score"
[[edge]]
from = "score"; to = "average"
[[edge]]
from = "average"; to = "play"
[[edge]]
from = "play"; to = "fall"; back = true; label = "next piece"

[[walk]]
title = "Look ahead — the falling piece"
text  = "The rulebook never reacts one piece at a time. It starts by listing every legal landing spot for the piece in play — every rotation in every column — and treats each one as a possible future board."
steps = ["fall"]
in    = { lang = "json", src = "data/flows/tetris_rulebook/01_in.json", label = "the recorded seed-607 clear turn" }
[[walk]]
title = "Chain the preview piece"
text  = "Each candidate board is paired with every landing spot of the preview piece — and the tree is pruned hard: only the best six boards survive each level, so the search stays tiny instead of exploding."
steps = ["preview"]
[[walk]]
title = "Cover the unknown third piece"
text  = "The piece after that is not known yet. Rather than gamble on a single guess, the plan plays out every piece the bag could still deal — the search branches over all of them."
steps = ["bag"]
[[walk]]
title = "Score every end board"
text  = "Each final board is scored by the strategy rulebook — holes, bumpiness, stack height, wells, lines cleared — the instincts of a careful human player, written as numbers."
steps = ["score"]
[[walk]]
title = "Average over the unknown · pick the plan"
text  = "A plan is only as good as its worst realistic draw: scores are averaged across the possible third pieces, so a line-clear that needs the I-piece rates low — it usually does not come. The sturdiest plan wins."
steps = ["average"]
out   = { lang = "json", src = "data/flows/tetris_rulebook/01_out.json", label = "per-spot scores · the winning plan" }
[[walk]]
title = "Play the first move — then re-plan"
text  = "Only the winning plan's first move is played; the very next piece restarts the whole search from scratch. All of it runs in about a third of a millisecond — a thousand-plus searches would fit inside one 60 Hz frame."
steps = ["play"]
edges = ["average->play", "play->fall"]
out   = { lang = "json", src = "data/flows/tetris_rulebook/02_out.json", label = "the recorded tetris landing" }
```

"Average over the unknown piece" is the point: a plan that only works if
the I-piece comes scores low, because it usually does not come.

## 5 · the rulebook's modes — play for tetrises only while it is safe

A cycle of back edges: BUILD branches into trouble, trouble hands back.

```gfflow
file  = "tetris_flow_modes.svg"
title = "The rulebook's modes: build for tetrises, dig or survive in trouble"
accent = "reflex"
intro = "Press play to walk the full build → trouble → recover cycle over the recorded seed-607 run. The board beside each step is the real recorded position."

[[lane]]
id = "build";   label = "Build";   note = "self-evolved weights";   color = "reflex"
[[lane]]
id = "trouble"; label = "Trouble"; note = "proven survival weights"; color = "reflex"

[[step]]
id = "bu"; n = "1"; lane = "build"; col = 0
title = "BUILD"; body = "nine flat + one open well; wait for tetrises"
status = "live"
[[step]]
id = "ds"; n = "2a"; lane = "trouble"; col = 1
title = "DOWNSTACK"; body = "covered holes — dig them back out first"
status = "live"
[[step]]
id = "sv"; n = "2b"; lane = "trouble"; col = 2
title = "SURVIVE"; body = "a tall stack — take any line, stay low"
status = "live"

[[edge]]
from = "bu"; to = "ds"; label = "3+ holes"
[[edge]]
from = "bu"; to = "sv"; label = "12+ rows"
[[edge]]
from = "ds"; to = "bu"; back = true; label = "holes dug out"
[[edge]]
from = "sv"; to = "bu"; back = true; label = "back under 12"

[[walk]]
title = "BUILD — the default mode"
text  = "On a clean, low stack it builds a 9-1 stack: nine columns packed flat plus one open well on the right edge, saving I-pieces to clear four lines at once. The 9-1 shape was self-evolved — nobody hand-coded it; it fell out of the training climb."
steps = ["bu"]
in    = { lang = "json", src = "data/flows/tetris_modes/01_in.json", label = "a real recorded build position" }
[[walk]]
title = "Trouble #1 — covered holes → DOWNSTACK"
text  = "Three or more holes buried under the stack and building stops paying: every new piece makes it worse. The mode flips to DOWNSTACK, which deliberately clears the lines sitting above the holes to dig them back out."
steps = ["ds"]
edges = ["bu->ds"]
in    = { lang = "json", src = "data/flows/tetris_modes/02_in.json", label = "the most-buried recorded position" }
out   = { lang = "json", src = "data/flows/tetris_modes/02_out.json", label = "the measured trigger vs the genome's threshold" }
[[walk]]
title = "Trouble #2 — tall stack → SURVIVE"
text  = "A stack twelve rows high is one bad piece from topping out. SURVIVE takes any line it can and keeps the board low — scoring gives way to staying alive. (Survive wins when both exits fire at once.)"
steps = ["sv"]
edges = ["bu->sv"]
in    = { lang = "json", src = "data/flows/tetris_modes/03_in.json", label = "the tallest recorded position" }
out   = { lang = "json", src = "data/flows/tetris_modes/03_out.json", label = "the measured height vs the genome's threshold" }
[[walk]]
title = "Recovery — both modes hand back to BUILD"
text  = "Holes dug out, or the stack back under twelve — either way the mode returns to BUILD and the cycle starts over. The board is re-read before every single piece, so the switch is never late."
steps = ["bu"]
edges = ["ds->bu", "sv->bu"]
in    = { lang = "json", src = "data/flows/tetris_modes/04_in.json", label = "a real recovered position" }
[[walk]]
title = "One search, three weights"
text  = "The modes are not three different AIs — it is the same placement search wearing different score weights: self-evolved builder weights while safe, proven survival weights in trouble. That is the whole trick behind the rulebook lane."
steps = ["bu", "ds", "sv"]
```

Build uses the self-evolved score weights (the 9-1 stack emerged from the
climb — nobody hand-coded it); Downstack and Survive use the proven
survival weights. The mode is re-read from the board before every piece
(Survive wins when both fire). Records: katgpt-rs `.benchmarks/892_tetris_rulebook_arena.md`.
