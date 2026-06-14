using Pkg
Pkg.activate(".")

include(joinpath(@__DIR__, "..", "src", "SORNMemory.jl"))

using .SORNMemory
using .SORNMemory.EpisodicMemory: create_episodic_memory, store!, recall!, get_stats
using .SORNMemory.Bridge: create_bridge, encode_tokens, decode_spikes, normalize_rates

function main()
    println("=" ^ 60)
    println("  SORN Memory — Pattern Learning Demo")
    println("=" ^ 60)
    println()

    mem = create_episodic_memory(
        n_exc=300, n_inh=75,
        vocab_size=100, embed_dim=16,
        neurons_per_token=8,
        timesteps_per_token=15,
        seed=42
    )

    println("Network created:")
    println("  Excitatory neurons: $(mem.sorn.exc.n)")
    println("  Inhibitory neurons: $(mem.sorn.inh.n)")
    println("  Input neurons: $(mem.bridge.n_input)")
    println("  Vocab size: $(mem.bridge.vocab_size)")
    println()

    sequences = [
        ([1, 2, 3, 4, 5], "A-B-C-D-E"),
        ([10, 20, 30], "X-Y-Z"),
        ([50, 51, 52, 53], "P-Q-R-S"),
    ]

    println("--- Phase 1: Learning sequences ---")
    println()

    for (seq, name) in sequences
        println("  Learning sequence '$name': $seq")
        for rep in 1:8
            store!(mem, seq)
        end
        println("    Stored 8 repetitions")
    end

    stats = get_stats(mem)
    println()
    println("After learning:")
    println("  Episodes: $(stats.n_episodes)")
    println("  Total stored: $(stats.total_stored)")
    println("  Network synapses: $(stats.n_synapses)")
    println()

    println("--- Phase 2: Recall test ---")
    println()

    test_cases = [
        ([1], "Should recall sequence starting with 1"),
        ([10], "Should recall sequence starting with 10"),
        ([50], "Should recall sequence starting with 50"),
    ]

    for (query, description) in test_cases
        println("  Query: $query")
        println("  Expected: $description")

        indices, scores = recall!(mem, query; top_k=8, method=:episode)

        println("    Recalled tokens: $indices (score: $(isempty(scores) ? 0.0 : round(scores[1], digits=4)))")
        println()
    end

    println("--- Phase 3: Online learning ---")
    println()

    println("  Storing 'hello' -> 'world' -> 'test'")
    store!(mem, [1])
    store!(mem, [2])
    store!(mem, [3])

    println("  Recalling with 'hello' (token 1)...")
    indices, scores = recall!(mem, [1]; top_k=5, method=:episode)

    println("    Recalled tokens: $indices (score: $(isempty(scores) ? 0.0 : round(scores[1], digits=4)))")
    println()

    stats = get_stats(mem)
    println("Final stats:")
    println("  Episodes: $(stats.n_episodes)")
    println("  Total stored: $(stats.total_stored)")
    println("  Avg importance: $(round(stats.avg_importance, digits=3))")
    println("  Avg recalls: $(round(stats.avg_recalls, digits=2))")
    println()
    println("=" ^ 60)
    println("  Pattern learning demo completed!")
    println("=" ^ 60)
end

main()
