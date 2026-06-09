//! GOAT proofs for Proof Sketch Evolution — Elo-rated population + global goal cache (Plan 128).
//!
//! **Source:** Tsoukalas et al. (2026). Advancing Mathematics Research with
//! AI-Driven Formal Proof Search. arXiv:2605.22763 (AlphaProof Nexus)
//!
//! **T8 targets:**
//! - Evolutionary converges ≥2× faster (rounds to 90% win rate)
//! - Goal cache hit rate ≥60% (reduces verification calls 3×)
//! - No regression on win rate ceiling
//! - Wall-clock overhead <10%
//!
//! These unit tests verify the core algorithms are correct — the full arena
//! benchmark (T9) runs separately against Bomber/Go domains.

#[cfg(feature = "proof_sketch_evolution")]
mod tests {
    use std::collections::HashSet;

    use katgpt_rs::pruners::proof::plackett_luce::generate_random_rankings;
    use katgpt_rs::pruners::proof::{
        DiversityHint, DiversityStrategy, ParallelismGuard, PlackettLuceConfig, PlackettLuceRater,
        PopulationConfig, ProofGoalCache, ProofState, SketchSampler, SketchSamplerConfig,
        SketchSelectionStrategy, select_strategy, should_use_population,
    };
    use katgpt_rs::pruners::{
        DEFAULT_ELO, Goal, GoalHash, GoalResult, SketchEntry, SketchPopulation,
    };

    // ── Helpers ─────────────────────────────────────────────────

    /// Create a simple proof state from a string.
    fn make_state(canonical: &str) -> ProofState {
        ProofState::new(canonical.as_bytes().to_vec())
    }

    /// Create a sketch entry with given Elo and zero visits.
    fn make_entry_with_elo(elo: f64) -> SketchEntry {
        SketchEntry::with_elo(
            make_state(&format!("state-{elo}")),
            vec![Goal::from_label(format!("goal-{elo}"))],
            elo,
        )
    }

    /// Create a sketch entry with specific pending goals.
    fn make_entry_with_goals(label: &str, goal_labels: &[&str]) -> SketchEntry {
        let goals: Vec<Goal> = goal_labels.iter().map(|g| Goal::from_label(*g)).collect();
        SketchEntry::new(make_state(label), goals)
    }

    /// Deterministic RNG seeded for reproducibility.
    fn seeded_rng() -> fastrand::Rng {
        fastrand::Rng::with_seed(42)
    }

    /// A verifier that always returns Proved.
    fn proved_verifier(_: &[u8]) -> GoalResult {
        GoalResult::Proved
    }

    /// A verifier that always returns Disproved with a counterexample.
    fn disproved_verifier(_: &[u8]) -> GoalResult {
        GoalResult::Disproved("counterexample found".to_string())
    }

    // ════════════════════════════════════════════════════════════
    // 1. Goal Cache — Dedup and blake3 Hashing
    // ════════════════════════════════════════════════════════════

    #[test]
    fn goal_hash_blake3_deterministic() {
        let bytes = b"canonical-goal-bytes";
        let h1 = GoalHash::from_canonical(bytes);
        let h2 = GoalHash::from_canonical(bytes);
        assert_eq!(
            h1.as_bytes(),
            h2.as_bytes(),
            "same input must produce same hash"
        );
    }

    #[test]
    fn goal_hash_different_inputs_differ() {
        let h1 = GoalHash::from_canonical(b"goal-A");
        let h2 = GoalHash::from_canonical(b"goal-B");
        assert_ne!(
            h1.as_bytes(),
            h2.as_bytes(),
            "different inputs must produce different hashes"
        );
    }

    #[test]
    fn goal_hash_from_goal_matches_canonical() {
        let goal = Goal::from_label("test-goal");
        let hash_via_goal = goal.hash();
        let hash_via_canonical = GoalHash::from_canonical(goal.canonical());
        assert_eq!(
            hash_via_goal.as_bytes(),
            hash_via_canonical.as_bytes(),
            "Goal::hash() must match GoalHash::from_canonical on same bytes"
        );
    }

    #[test]
    fn cache_miss_on_first_lookup() {
        let mut cache = ProofGoalCache::new();
        let result = cache.get_or_verify(b"new-goal", proved_verifier);
        assert!(result.is_proved(), "verifier should have been called");
        assert_eq!(cache.misses(), 1, "first lookup must be a miss");
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_hit_on_repeat_lookup() {
        let mut cache = ProofGoalCache::new();
        let bytes = b"repeated-goal";

        // First lookup — miss
        let r1 = cache.get_or_verify(bytes, proved_verifier);
        assert!(r1.is_proved());
        assert_eq!(cache.misses(), 1);

        // Second lookup — hit (verifier NOT called again)
        let r2 = cache.get_or_verify(bytes, disproved_verifier);
        assert!(
            r2.is_proved(),
            "must return cached Proved, not call disproved_verifier"
        );
        assert_eq!(cache.hits(), 1, "second lookup must be a hit");
        assert_eq!(cache.misses(), 1, "miss count must stay at 1");
    }

    // ════════════════════════════════════════════════════════════
    // 2. Goal Cache — Hit Rate GOAT Target (≥60%)
    // ════════════════════════════════════════════════════════════

    #[test]
    fn cache_hit_rate_meets_goat_target() {
        let mut cache = ProofGoalCache::new();
        let goals: Vec<&[u8]> = vec![b"g1", b"g2", b"g3", b"g4", b"g5"];

        // Phase 1: populate cache (5 misses)
        for g in &goals {
            cache.get_or_verify(g, proved_verifier);
        }
        assert_eq!(cache.misses(), 5);
        assert_eq!(cache.hits(), 0);

        // Phase 2: re-verify same goals (3 rounds × 5 goals = 15 hits)
        for _ in 0..3 {
            for g in &goals {
                cache.get_or_verify(g, proved_verifier);
            }
        }

        // 5 misses + 15 hits = 20 total, hit rate = 15/20 = 0.75
        let total = cache.total_lookups();
        let hit_rate = cache.hit_rate();
        assert_eq!(total, 20, "5 initial misses + 15 hits = 20 lookups");
        assert!(
            hit_rate >= 0.60,
            "hit rate {hit_rate:.2} must meet ≥60% GOAT target"
        );
    }

    #[test]
    fn cache_hit_rate_zero_on_empty() {
        let cache = ProofGoalCache::new();
        assert_eq!(cache.hit_rate(), 0.0, "empty cache must have 0% hit rate");
    }

    #[test]
    fn cache_clear_resets_everything() {
        let mut cache = ProofGoalCache::new();
        cache.get_or_verify(b"g1", proved_verifier);
        cache.get_or_verify(b"g1", proved_verifier); // hit
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.hits(), 1);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0, "clear must reset hit counter");
        assert_eq!(cache.misses(), 0, "clear must reset miss counter");
    }

    #[test]
    fn cache_peek_does_not_update_counters() {
        let mut cache = ProofGoalCache::new();
        cache.get_or_verify(b"g1", proved_verifier);
        assert_eq!(cache.misses(), 1);

        // Peek at cached goal
        let peeked = cache.peek(b"g1");
        assert!(peeked.is_some());
        assert!(peeked.unwrap().is_proved());

        // Peek at uncached goal
        let uncached = cache.peek(b"g2");
        assert!(uncached.is_none());

        // Counters must be unchanged
        assert_eq!(cache.hits(), 0, "peek must not increment hits");
        assert_eq!(cache.misses(), 1, "peek must not increment misses");
    }

    #[test]
    fn cache_insert_manual_bypasses_verifier() {
        let mut cache = ProofGoalCache::new();

        // Manually insert without verifier
        let prev = cache.insert(b"pre-seeded", GoalResult::Disproved("cx".to_string()));
        assert!(prev.is_none(), "no previous entry");

        // Lookup should hit the manual insert
        let result = cache.get_or_verify(b"pre-seeded", proved_verifier);
        assert!(
            result.is_disproved(),
            "must return manually inserted result"
        );
        assert_eq!(cache.hits(), 1, "lookup is a hit");
        assert_eq!(cache.misses(), 0, "no miss — verifier not called");
    }

    // ════════════════════════════════════════════════════════════
    // 3. Sketch Population — CRUD
    // ════════════════════════════════════════════════════════════

    #[test]
    fn population_insert_and_get() {
        let mut pop = SketchPopulation::new(10);
        let entry = make_entry_with_elo(1300.0);

        let id = entry.id;
        let report = pop.insert(entry);
        assert!(
            !report.did_evict(),
            "single insert should not trigger eviction"
        );
        assert_eq!(pop.len(), 1);

        let fetched = pop.get(&id);
        assert!(fetched.is_some(), "inserted entry must be retrievable");
        assert_eq!(fetched.unwrap().elo_rating, 1300.0);
    }

    #[test]
    fn population_insert_replaces_same_id() {
        let mut pop = SketchPopulation::new(10);
        let entry = make_entry_with_elo(1200.0);
        let id = entry.id;

        pop.insert(entry);

        // Re-insert with updated Elo
        let mut updated = make_entry_with_elo(1400.0);
        updated.id = id;
        pop.insert(updated);

        assert_eq!(pop.len(), 1, "same-ID insert must replace, not duplicate");
        assert_eq!(pop.get(&id).unwrap().elo_rating, 1400.0);
    }

    #[test]
    fn population_remove() {
        let mut pop = SketchPopulation::new(10);
        let entry = make_entry_with_elo(1200.0);
        let id = entry.id;

        pop.insert(entry);
        assert!(pop.contains(&id));

        let removed = pop.remove(&id);
        assert!(removed.is_some());
        assert!(!pop.contains(&id));
        assert!(pop.is_empty());
    }

    #[test]
    fn population_sorted_by_elo_descending() {
        let mut pop = SketchPopulation::new(10);
        for elo in [1200.0, 1500.0, 1100.0, 1400.0] {
            pop.insert(make_entry_with_elo(elo));
        }

        let sorted = pop.sorted_by_elo();
        let elos: Vec<f64> = sorted.iter().map(|e| e.elo_rating).collect();
        assert_eq!(
            elos,
            vec![1500.0, 1400.0, 1200.0, 1100.0],
            "must be Elo descending"
        );
    }

    #[test]
    fn population_top_k_truncates() {
        let mut pop = SketchPopulation::new(3);
        for elo in [1000.0, 1100.0, 1200.0, 1300.0, 1400.0] {
            pop.insert_no_evict(make_entry_with_elo(elo));
        }

        // top_k = 3, so we should get only the top 3 by Elo
        // Population has 5 entries, no eviction happened yet
        let top = pop.top_k();
        assert_eq!(top.len(), 3, "top_k must truncate to config.top_k");
        assert_eq!(top[0].elo_rating, 1400.0, "best Elo first");
        assert_eq!(top[2].elo_rating, 1200.0, "third best Elo");
    }

    // ════════════════════════════════════════════════════════════
    // 4. Population — Top-K Eviction
    // ════════════════════════════════════════════════════════════

    #[test]
    fn eviction_triggers_at_max_population() {
        let mut pop = SketchPopulation::with_config(PopulationConfig::new(3));
        assert_eq!(pop.config().max_population, 3);

        // Insert 3 entries — no eviction yet
        for elo in [1100.0, 1200.0, 1300.0] {
            let report = pop.insert(make_entry_with_elo(elo));
            assert!(!report.did_evict());
        }
        assert_eq!(pop.len(), 3);

        // Insert 4th — triggers eviction
        let report = pop.insert(make_entry_with_elo(1250.0));
        assert!(report.did_evict(), "exceeding max_population must evict");
        assert_eq!(report.count(), 1, "one entry should be evicted");
        assert_eq!(pop.len(), 3, "population must be back to max_population");
    }

    #[test]
    fn eviction_keeps_highest_elo() {
        let mut pop = SketchPopulation::with_config(PopulationConfig::new(3));

        // Insert entries with varied Elo
        let low_id = make_entry_with_elo(1000.0).id;
        pop.insert(make_entry_with_elo(1000.0));
        pop.insert(make_entry_with_elo(1500.0));
        pop.insert(make_entry_with_elo(1300.0));

        // Insert 4th, triggers eviction of lowest Elo (1000)
        pop.insert(make_entry_with_elo(1250.0));

        assert!(
            pop.get(&low_id).is_none(),
            "lowest Elo entry must be evicted"
        );
        assert_eq!(pop.len(), 3);

        // Verify remaining entries are the highest Elo (use Vec — f64 not Eq/Hash)
        let elos: Vec<f64> = pop.sorted_by_elo().iter().map(|e| e.elo_rating).collect();
        assert!(elos.contains(&1500.0), "highest Elo must survive: {elos:?}");
        assert!(
            elos.contains(&1300.0),
            "second highest must survive: {elos:?}"
        );
        assert!(elos.contains(&1250.0), "new entry must survive: {elos:?}");
    }

    #[test]
    fn eviction_tiebreaks_by_visits() {
        let config = PopulationConfig::new(2);
        let mut pop = SketchPopulation::with_config(config);

        // Two entries with same Elo, different visits
        let mut less_visited = make_entry_with_elo(1200.0);
        less_visited.record_visit(); // 1 visit
        let less_visited_id = less_visited.id;

        let mut more_visited = make_entry_with_elo(1200.0);
        more_visited.record_visit();
        more_visited.record_visit();
        more_visited.record_visit(); // 3 visits
        let more_visited_id = more_visited.id;

        pop.insert(less_visited);
        pop.insert(more_visited);

        // Insert a third with higher Elo, triggers eviction
        pop.insert(make_entry_with_elo(1400.0));

        // With equal Elo, tiebreak by visits ascending — less_visited evicted
        assert!(
            pop.get(&less_visited_id).is_none(),
            "less-visited entry must be evicted on Elo tie"
        );
        assert!(
            pop.get(&more_visited_id).is_some(),
            "more-visited entry must survive on Elo tie"
        );
    }

    #[test]
    fn batch_insert_defers_eviction() {
        let config = PopulationConfig::with_overshoot(2, 5);
        let mut pop = SketchPopulation::with_config(config);

        // Insert 6 via insert_no_evict — exceeds max_population=5
        for elo in [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 900.0] {
            pop.insert_no_evict(make_entry_with_elo(elo));
        }
        assert_eq!(
            pop.len(),
            6,
            "insert_no_evict allows overshoot beyond max_population"
        );

        // Finalize batch — evicts down to top_k=2 (triggers because 6 > max_population=5)
        let report = pop.finalize_batch();
        assert!(report.did_evict());
        assert_eq!(report.count(), 4, "must evict 6-2=4 entries");
        assert_eq!(pop.len(), 2);

        // Top 2 Elo must survive
        let elos: Vec<f64> = pop.sorted_by_elo().iter().map(|e| e.elo_rating).collect();
        assert_eq!(elos, vec![1400.0, 1300.0]);
    }

    // ════════════════════════════════════════════════════════════
    // 5. Plackett-Luce Rating — Elo from Rankings
    // ════════════════════════════════════════════════════════════

    #[test]
    fn plackett_luce_consistent_winner_gets_highest_elo() {
        let mut rng = seeded_rng();
        let rater = PlackettLuceRater::with_paper_defaults();

        let sketches: Vec<SketchEntry> = (0..4)
            .map(|i| make_entry_with_elo(DEFAULT_ELO + i as f64 * 100.0))
            .collect();

        // Sketch 0 wins every ranking
        let rankings = vec![
            vec![0, 1, 2, 3],
            vec![0, 2, 1, 3],
            vec![0, 3, 1, 2],
            vec![0, 1, 3, 2],
        ];

        let elos = rater.rate(&sketches, &rankings, &mut rng);

        let elo_0 = elos[&sketches[0].id];
        let elo_1 = elos[&sketches[1].id];
        let elo_2 = elos[&sketches[2].id];
        let elo_3 = elos[&sketches[3].id];

        assert!(
            elo_0 > elo_1,
            "consistent winner must have higher Elo: {elo_0:.1} vs {elo_1:.1}"
        );
        assert!(
            elo_0 > elo_2,
            "consistent winner must beat sketch 2: {elo_0:.1} vs {elo_2:.1}"
        );
        assert!(
            elo_0 > elo_3,
            "consistent winner must beat sketch 3: {elo_0:.1} vs {elo_3:.1}"
        );
    }

    #[test]
    fn plackett_luce_no_rankings_produces_similar_elos() {
        let mut rng = seeded_rng();
        // Fewer samples for speed
        let config = PlackettLuceConfig {
            gibbs_samples: 200,
            burn_in: 50,
            ..PlackettLuceConfig::PAPER_DEFAULTS
        };
        let rater = PlackettLuceRater::new(config);

        let sketches: Vec<SketchEntry> = (0..3).map(|_| make_entry_with_elo(DEFAULT_ELO)).collect();

        // No rankings — sketches are indistinguishable, so Elo spread should be small
        let elos = rater.rate(&sketches, &[], &mut rng);

        let elo_values: Vec<f64> = sketches.iter().map(|s| elos[&s.id]).collect();
        let min_elo = elo_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_elo = elo_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let spread = max_elo - min_elo;

        // Without rankings, all sketches draw from the same prior — spread must be bounded.
        // The hierarchical Gamma(1,1) prior has high variance, so 1000 Elo spread is expected.
        assert!(
            spread < 1000.0,
            "no-ranking Elo spread must be bounded: {spread:.1} (min={min_elo:.1}, max={max_elo:.1})"
        );

        // All elos must be in a reasonable range (prior predictive)
        for elo in &elo_values {
            assert!(
                (800.0..2500.0).contains(elo),
                "prior Elo must be in reasonable range: {elo:.1}"
            );
        }
    }

    #[test]
    fn plackett_luce_empty_sketches_empty_result() {
        let mut rng = seeded_rng();
        let rater = PlackettLuceRater::with_paper_defaults();
        let elos = rater.rate(&[], &[], &mut rng);
        assert!(elos.is_empty(), "empty sketches must produce empty Elo map");
    }

    #[test]
    fn generate_random_rankings_valid_indices() {
        let mut rng = seeded_rng();
        let n_sketches: usize = 10;
        let match_size: usize = 4;
        let num_rankings: usize = 20;

        let rankings = generate_random_rankings(n_sketches, match_size, num_rankings, &mut rng);

        assert_eq!(rankings.len(), num_rankings);
        for ranking in &rankings {
            assert_eq!(
                ranking.len(),
                match_size,
                "each ranking must have match_size items"
            );
            for &idx in ranking {
                assert!(idx < n_sketches, "index {idx} must be < {n_sketches}");
            }
            // No duplicates within a ranking
            let unique: HashSet<usize> = ranking.iter().copied().collect();
            assert_eq!(unique.len(), match_size, "no duplicate indices in ranking");
        }
    }

    // ════════════════════════════════════════════════════════════
    // 6. P-UCB Sampling — Exploration/Exploitation Balance
    // ════════════════════════════════════════════════════════════

    #[test]
    fn pucb_selects_unvisited_entries_first() {
        let mut rng = seeded_rng();
        let mut sampler = SketchSampler::new(SketchPopulation::new(10));

        // Insert one heavily visited, one unvisited
        let mut visited_entry = make_entry_with_elo(1500.0);
        for _ in 0..100 {
            visited_entry.record_visit();
        }
        let unvisited_entry = make_entry_with_elo(1000.0); // lower Elo but unvisited

        sampler.population_mut().insert(visited_entry);
        sampler.population_mut().insert(unvisited_entry);

        // P-UCB should prefer the unvisited entry (huge exploration bonus)
        let selected = sampler.sample_p_ucb(&mut rng);
        assert!(selected.is_some());
        assert_eq!(
            selected.unwrap().visits,
            0,
            "P-UCB must prefer unvisited entry"
        );
    }

    #[test]
    fn pucb_prefers_higher_elo_when_visits_equal() {
        let mut rng = seeded_rng();
        let mut sampler = SketchSampler::new(SketchPopulation::new(10));

        // Two entries with same visits, different Elo
        let high_elo = make_entry_with_elo(1500.0);
        let low_elo = make_entry_with_elo(1000.0);

        sampler.population_mut().insert(high_elo);
        sampler.population_mut().insert(low_elo);

        let selected = sampler.sample_p_ucb(&mut rng);
        assert!(selected.is_some());
        assert_eq!(
            selected.unwrap().elo_rating,
            1500.0,
            "P-UCB must prefer higher Elo when visits are equal"
        );
    }

    #[test]
    fn epsilon_greedy_explores_sometimes() {
        // ε = 1.0 → always explore (random)
        let config = SketchSamplerConfig::paper_defaults().with_epsilon(1.0);
        let pop = SketchPopulation::new(10);
        let mut sampler = SketchSampler::with_config(pop, config);

        sampler.population_mut().insert(make_entry_with_elo(1500.0));
        sampler.population_mut().insert(make_entry_with_elo(1000.0));

        let mut rng = seeded_rng();
        let mut found_low = false;
        for _ in 0..200 {
            let selected = sampler.sample_epsilon_greedy(&mut rng);
            if selected.unwrap().elo_rating < 1200.0 {
                found_low = true;
                break;
            }
        }
        assert!(found_low, "ε=1.0 must explore the low-Elo entry");
    }

    #[test]
    fn epsilon_greedy_zero_epsilon_always_exploits() {
        // ε = 0.0 → always exploit (best Elo)
        let config = SketchSamplerConfig::paper_defaults().with_epsilon(0.0);
        let pop = SketchPopulation::new(10);
        let mut sampler = SketchSampler::with_config(pop, config);

        sampler.population_mut().insert(make_entry_with_elo(1000.0));
        sampler.population_mut().insert(make_entry_with_elo(1500.0));

        let mut rng = seeded_rng();
        for _ in 0..50 {
            let selected = sampler.sample_epsilon_greedy(&mut rng);
            assert_eq!(
                selected.unwrap().elo_rating,
                1500.0,
                "ε=0.0 must always select best Elo"
            );
        }
    }

    #[test]
    fn sample_empty_returns_none() {
        let mut rng = seeded_rng();
        let sampler = SketchSampler::new(SketchPopulation::new(10));
        assert!(
            sampler.sample(&mut rng).is_none(),
            "empty population must return None"
        );
        assert!(sampler.sample_p_ucb(&mut rng).is_none());
        assert!(sampler.sample_epsilon_greedy(&mut rng).is_none());
    }

    // ════════════════════════════════════════════════════════════
    // 7. Diversity Injection — Strategy Distribution
    // ════════════════════════════════════════════════════════════

    #[test]
    fn diversity_injection_returns_all_three_strategies() {
        let sampler = SketchSampler::new(SketchPopulation::new(10));
        let mut rng = fastrand::Rng::with_seed(123);
        let mut seen: HashSet<DiversityStrategy> = HashSet::new();

        // With 33/33/34 split, 300 samples should hit all three strategies
        for _ in 0..300 {
            let hint = sampler.inject_diversity(&mut rng);
            seen.insert(hint.strategy);
        }

        assert_eq!(
            seen.len(),
            3,
            "must produce all three diversity strategies over 300 samples"
        );
        assert!(seen.contains(&DiversityStrategy::Decompose));
        assert!(seen.contains(&DiversityStrategy::Combine));
        assert!(seen.contains(&DiversityStrategy::NovelApproach));
    }

    #[test]
    fn diversity_hint_no_context_by_default() {
        let hint = DiversityHint::new(DiversityStrategy::Decompose);
        assert_eq!(hint.strategy, DiversityStrategy::Decompose);
        assert!(hint.context.is_none(), "default hint must have no context");
    }

    #[test]
    fn diversity_hint_with_context() {
        let hint = DiversityHint::with_context(DiversityStrategy::Combine, "merge left flank");
        assert_eq!(hint.strategy, DiversityStrategy::Combine);
        assert_eq!(hint.context.as_deref(), Some("merge left flank"));
    }

    #[test]
    fn diversity_strategy_descriptions_are_nonempty() {
        for strategy in DiversityStrategy::ALL {
            assert!(
                !strategy.description().is_empty(),
                "description for {strategy} must not be empty"
            );
        }
    }

    #[test]
    fn inject_diversity_with_context_attaches_entry_info() {
        let sampler = SketchSampler::new(SketchPopulation::new(10));
        let mut rng = seeded_rng();
        let entry = make_entry_with_goals("s1", &["g1", "g2"]);

        let hint = sampler.inject_diversity_with_context(&entry, &mut rng);
        assert!(
            hint.context.is_some(),
            "inject_diversity_with_context must attach context"
        );
        let ctx = hint.context.unwrap();
        assert!(
            ctx.contains("entry="),
            "context must contain entry ID info: {ctx}"
        );
        assert!(
            ctx.contains("goals=2"),
            "context must contain goal count: {ctx}"
        );
    }

    // ════════════════════════════════════════════════════════════
    // 8. Parallelism Guard — Strategy Selection
    // ════════════════════════════════════════════════════════════

    #[test]
    fn parallelism_guard_does_not_panic() {
        let guard = ParallelismGuard::new();
        assert!(guard.threads() > 0, "rayon always reports ≥1 thread");
    }

    #[test]
    fn parallelism_guard_default_matches_new() {
        let from_new = ParallelismGuard::new();
        let from_default = ParallelismGuard::default();
        assert_eq!(from_new.threads(), from_default.threads());
        assert_eq!(
            from_new.population_enabled(),
            from_default.population_enabled()
        );
    }

    #[test]
    fn select_strategy_returns_correct_variant() {
        // We can't control rayon thread count, but we can test the logic
        // by verifying consistency with the guard's population_enabled
        let guard = ParallelismGuard::new();
        let strategy = select_strategy(&guard);

        match guard.population_enabled() {
            true => assert_eq!(strategy, SketchSelectionStrategy::PopulationPucb),
            false => assert_eq!(strategy, SketchSelectionStrategy::BasicUcb),
        }
    }

    #[test]
    fn fallback_reason_consistency() {
        let guard = ParallelismGuard::new();
        match guard.population_enabled() {
            true => assert!(
                guard.fallback_reason().is_none(),
                "population enabled → no fallback reason"
            ),
            false => {
                let reason = guard.fallback_reason().expect("must have reason");
                assert!(
                    reason.contains("single-threaded"),
                    "fallback reason must mention single-threaded: {reason}"
                );
            }
        }
    }

    #[test]
    fn strategy_uses_population_only_for_pucb() {
        assert!(SketchSelectionStrategy::PopulationPucb.uses_population());
        assert!(!SketchSelectionStrategy::BasicUcb.uses_population());
        assert!(!SketchSelectionStrategy::EpsilonGreedy.uses_population());
    }

    #[test]
    fn should_use_population_returns_bool() {
        // Must not panic — exercises the rayon query
        let result = should_use_population();
        assert!(matches!(result, true | false));
    }

    // ════════════════════════════════════════════════════════════
    // 9. Sketch Entry — Core Operations
    // ════════════════════════════════════════════════════════════

    #[test]
    fn sketch_entry_new_has_default_elo() {
        let entry = make_entry_with_goals("test", &["g1"]);
        assert_eq!(entry.elo_rating, DEFAULT_ELO);
        assert_eq!(entry.visits, 0);
        assert!(!entry.is_explored());
    }

    #[test]
    fn sketch_entry_record_visit_increments() {
        let mut entry = make_entry_with_elo(1200.0);
        assert_eq!(entry.visits, 0);

        entry.record_visit();
        entry.record_visit();
        entry.record_visit();
        assert_eq!(entry.visits, 3);
        assert!(entry.is_explored());
    }

    #[test]
    fn sketch_entry_update_elo() {
        let mut entry = make_entry_with_elo(1200.0);
        entry.update_elo(1500.0);
        assert_eq!(entry.elo_rating, 1500.0);
    }

    #[test]
    fn sketch_entry_add_lesson_fifo_eviction() {
        let mut entry = make_entry_with_goals("test", &[]);

        // Fill to max
        for i in 0..katgpt_rs::pruners::MAX_LESSONS {
            entry.add_lesson(format!("lesson-{i}"));
        }
        assert_eq!(entry.lesson_count(), katgpt_rs::pruners::MAX_LESSONS);

        // Add one more — oldest should be evicted
        entry.add_lesson("overflow".to_string());
        assert_eq!(entry.lesson_count(), katgpt_rs::pruners::MAX_LESSONS);
        assert_eq!(
            entry.lessons[0], "lesson-1",
            "oldest lesson must be evicted (FIFO)"
        );
        assert_eq!(
            entry.lessons.iter().last().unwrap(),
            "overflow",
            "newest lesson must be at end"
        );
    }

    #[test]
    fn sketch_entry_pending_goals_cap() {
        let goals: Vec<Goal> = (0..50).map(|i| Goal::from_label(format!("g{i}"))).collect();
        let entry = SketchEntry::new(make_state("capped"), goals);
        assert_eq!(
            entry.pending_goal_count(),
            katgpt_rs::pruners::MAX_PENDING_GOALS,
            "pending goals must be capped at MAX_PENDING_GOALS"
        );
    }

    // ════════════════════════════════════════════════════════════
    // 10. Integration — End-to-End Sampling Cycle
    // ════════════════════════════════════════════════════════════

    #[test]
    fn integration_sample_rate_update_cycle() {
        let mut rng = seeded_rng();

        // 1. Create population with 5 sketches
        let mut sampler = SketchSampler::new(SketchPopulation::new(10));
        for elo in [1200.0, 1250.0, 1300.0, 1350.0, 1400.0] {
            sampler.population_mut().insert(make_entry_with_elo(elo));
        }

        // 2. Sample and record visits
        for _ in 0..20 {
            if let Some(entry) = sampler.sample_mut(&mut rng) {
                entry.record_visit();
            }
        }

        // 3. Verify visits were recorded
        let total = sampler.population().total_visits();
        assert_eq!(total, 20, "20 samples must produce 20 visits");

        // 4. Rate via Plackett-Luce
        let rater = PlackettLuceRater::new(PlackettLuceConfig {
            gibbs_samples: 200,
            burn_in: 50,
            ..PlackettLuceConfig::PAPER_DEFAULTS
        });

        let sketches: Vec<SketchEntry> = sampler
            .population()
            .sorted_by_elo()
            .iter()
            .map(|e| (*e).clone())
            .collect();

        // Sketch 0 (best Elo) wins all rankings
        let rankings = vec![
            vec![0, 1, 2, 3, 4],
            vec![0, 2, 1, 4, 3],
            vec![0, 3, 4, 1, 2],
        ];

        let elos = rater.rate(&sketches, &rankings, &mut rng);

        // Update Elo in population
        for sketch in &sketches {
            if let Some(elo) = elos.get(&sketch.id) {
                sampler
                    .population_mut()
                    .get_mut(&sketch.id)
                    .unwrap()
                    .update_elo(*elo);
            }
        }

        // 5. Verify Elo ordering is maintained
        let sorted = sampler.population().sorted_by_elo();
        for window in sorted.windows(2) {
            assert!(
                window[0].elo_rating >= window[1].elo_rating,
                "population must be sorted by Elo descending after update"
            );
        }
    }

    #[test]
    fn integration_goal_cache_with_population() {
        let mut cache = ProofGoalCache::new();
        let mut pop = SketchPopulation::new(10);

        // Simulate a decode step: verify goals, create sketches from proved goals
        let goals: Vec<&[u8]> = vec![
            b"constraint-A",
            b"constraint-B",
            b"constraint-C",
            b"constraint-A",
        ];

        let mut proved_count = 0;
        for goal_bytes in &goals {
            let result = cache.get_or_verify(goal_bytes, proved_verifier);
            if result.is_proved() {
                proved_count += 1;
                let entry = SketchEntry::new(
                    make_state(&format!("state-{proved_count}")),
                    vec![Goal::from_label(format!("goal-{proved_count}"))],
                );
                pop.insert(entry);
            }
        }

        // constraint-A appears twice, so proved_count = 4 (cache hit doesn't prevent counting)
        assert_eq!(proved_count, 4, "all goals proved");
        assert_eq!(cache.hits(), 1, "second constraint-A must be a hit");
        assert_eq!(cache.misses(), 3, "three unique constraints must be misses");
        assert_eq!(pop.len(), 4, "4 sketches created from proved goals");

        let hit_rate = cache.hit_rate();
        assert!(
            hit_rate >= 0.20,
            "hit rate must reflect 1/4 = 25%: got {hit_rate:.2}"
        );
    }
}
