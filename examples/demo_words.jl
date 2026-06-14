using SORNMemory
using Statistics: mean

println("=" ^ 60)
println("  FRP Neural Memory — Real Word Demo")
println("=" ^ 60)
println()

tok = Tokenizer.Tokenizer()
println("Tokenizer created")
println()

# Use encode to add words and get IDs
function ensure_ids(tok, words)
    # encode adds words to vocab if space available
    text = join(words, " ")
    ids = encode(tok, text)
    println("  Words: $(join(words, ", ")) -> IDs: $ids")
    ids
end

seed = 42

# Create memory system
mem = create_episodic_memory(n_exc=300, n_inh=75,
                             vocab_size=1000, embed_dim=32,
                             timesteps_per_token=20, max_episodes=50,
                             seed=seed)

println("--- STORING EPISODES ---")

ep1 = ensure_ids(tok, ["cat", "eat", "fish"])
store!(mem, ep1)

ep2 = ensure_ids(tok, ["dog", "run", "fast"])
store!(mem, ep2)

ep3 = ensure_ids(tok, ["big", "red", "apple"])
store!(mem, ep3)

ep4 = ensure_ids(tok, ["bird", "fly", "blue", "sky"])
store!(mem, ep4)

ep5 = ensure_ids(tok, ["horse", "run", "fast", "farm"])
store!(mem, ep5)

ep6 = ensure_ids(tok, ["small", "fish", "swim", "sea"])
store!(mem, ep6)

println("  Total episodes: $(length(mem.episodes))")
println()

# Train readouts
println("--- TRAINING READOUTS (ridge regression, freeze! protocol) ---")
train_readout!(mem; alpha=1.0)
println("  SORN readout: $(size(mem.readout.projection))")
println("  FRP readout:  $(size(mem.frp_readout.projection))")
println()

# Helper to decode token IDs
function decode_words(tok, ids)
    words = decode(tok, ids)
    filter!(w -> w != "<UNK>", words)
    join(words, ", ")
end

# Test queries
test_queries = [
    "cat fish",
    "dog run",
    "red apple",
    "bird fly",
    "horse farm",
    "blue sea",
]

println("--- NEURAL DECODE (FRP, method=:neural) ---")
println()
for desc in test_queries
    tids = encode(tok, desc)
    println("Query: \"$desc\" -> $(decode_words(tok, tids)) (ids=$tids)")
    ids, scores = recall!(mem, tids; top_k=5, method=:neural)
    words_decoded = decode(tok, ids)
    println("  Decoded: $(join(words_decoded, ", "))")
    println("  Scores:  $(join(string.(round.(scores, digits=4)), ", "))")
    println()
end

println("--- EPISODE RECALL (method=:episode, for comparison) ---")
println()
for desc in test_queries
    tids = encode(tok, desc)
    ids, scores = recall!(mem, tids; top_k=3, method=:episode)
    words_decoded = decode(tok, ids)
    println("Query: \"$desc\"")
    println("  Recalled: $(join(words_decoded, " "))")
    println()
end

println("--- SUMMARY ---")
println("  Episodes stored: $(length(mem.episodes))")
println("  SORN mean |W_EE|: $(round(mean(abs.(mem.sorn.W_EE.nzval)), digits=4))")
println("  FRP: fixed random projection (no plasticity)")
println()
println("Note: Fully deterministic with the same seed. encode_tokens uses a seeded")
println("MersenneTwister (from mem.rng_seed), so Poisson spike trains are reproducible.")
