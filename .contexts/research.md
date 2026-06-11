https://r.jina.ai/https://arxiv.org/pdf/2606.07878v1

0. distill idea to our modelless [@katgpt-rs-core](file:///Users/katopz/git/katgpt-rs/crates/katgpt-rs-core/) [@README.md](file:///Users/katopz/git/katgpt-rs/README.md) [@src](file:///Users/katopz/git/katgpt-rs/src/)
and/or model based:
[@README.md](file:///Users/katopz/git/riir-ai/README.md) [@crates](file:///Users/katopz/git/riir-ai/crates/)

1. be createive, think fundamental apply not just direct mapping to create new fusion novel idea. grep related in research, verdict goat/gain and make dicision by [@003_Commercial_Open_Source_Strategy_Verdict.md](file:///Users/katopz/git/katgpt-rs/.research/003_Commercial_Open_Source_Strategy_Verdict.md) to create research md at
modelless: /Users/katopz/git/katgpt-rs/.research
and/or model based: /Users/katopz/git/riir-ai/.research

2. if gain do add plan md to modelless /Users/katopz/git/katgpt-rs/.plans and/or model based /Users/katopz/git/riir-ai/.plans with `## Task and - [ ]` to each task to indicate task status

3. Try implement to unblock if block, after check goat + proof gain, promote to default if gain also demote loser.

4. If curious do mcp web search and web reader or keyword search e.g.: 
https://r.jina.ai/https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term={YOUR_KEYWORD_HERE}&terms-0-field=abstract&classification-computer_science=y&classification-mathematics=y&classification-physics_archives=all&classification-statistics=y&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first

constraints:

1. **Modelless first** — inference-time only, no LLM training
2. **Land in riir-ai domain** — keep the commercial strategy (engine/fuel split) intact
3. **LoRA only for training** — no full LLM training, closest is freeze/thaw weight dumps
4. **Self-learning adaptive CoT welcome** — but no LLM training
5. **SOLID, DRY** — per [@optimization.md](file:///Users/katopz/git/katgpt-rs/.contexts/optimization.md)
6. **Tests/examples** showing before/after thinking vs non-thinking with expected gains
7. **CPU/GPU auto-route** when load changes
8. mind plasma/hot/warm/cold/freeze path, aim for both perf for game and more sec for chain
9. use threhold to adpative between cpu/simd/gpu/ane if need
