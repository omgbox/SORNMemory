using SORNMemory
using Statistics: mean
using Random: MersenneTwister

println("=" ^ 65)
println("  SORN Temporal Dynamics — Real Word Demo")
println("=" ^ 65)
println()

tok = Tokenizer.Tokenizer()
mem = create_episodic_memory(seed=42)

# Store episodes
episodes = [
    "the cat eats fish in the kitchen",
    "a dog runs fast through the park",
    "i ate a big red apple for lunch",
    "a blue bird flies high in the sky",
    "the horse runs fast on the farm",
    "a small fish swims in the blue sea",
]
for text in episodes
    ids = encode(tok, text)
    store!(mem, ids)
end
train_readout!(mem; alpha=1.0)
println("Stored $(length(episodes)) episodes")
println()

# Show SORN evolves firing rates differently per token
println("--- SORN firing rates per token position ---")
println()

test_queries = [
    "cat fish",
    "bird sky",
    "dog park",
    "horse farm",
]
n_bins = 4
n_exc = mem.sorn.exc.n

for query in test_queries
    ids = encode(tok, query)
    spikes = SORNMemory.Bridge.encode_tokens(mem.bridge, ids; timesteps_per_token=20, rng=MersenneTwister(mem.rng_seed))
    result = SORNMemory.SNN.Simulation.simulate!(mem.sorn, spikes; verbose=false)

    println("Query: \"$query\"")
    for pos in 1:length(ids)
        t_start = (pos - 1) * 20 + 1
        t_end = pos * 20
        # Average E rate over this token window
        e_spikes = result.spikes[1:n_exc, t_start:t_end]
        token_rate = mean(e_spikes)
        # Per-neuron firing rates for first 10 neurons
        first10 = [round(mean(result.spikes[i, t_start:t_end]), digits=3) for i in 1:min(10, n_exc)]
        println("  Token $(pos) (id=$(ids[pos])): avg E rate=$(round(token_rate*100, digits=1))%")
        println("    First 10 neuron rates: $(join(first10, ", "))")
    end
    println()
end

# Show temporal profile (4-bin) differences between tokens
println("--- Temporal profiles: SORN distinguishes tokens via fine structure ---")
println()

# Compare "cat" vs "dog" profiles
cat_ids = encode(tok, "cat eats fish")
dog_ids = encode(tok, "dog runs fast")
all_ids = encode(tok, "cat eats fish the dog runs fast")

spikes = SORNMemory.Bridge.encode_tokens(mem.bridge, all_ids; timesteps_per_token=20, rng=MersenneTwister(mem.rng_seed))
result = SORNMemory.SNN.Simulation.simulate!(mem.sorn, spikes; verbose=false)

println("Comparing first 3 neurons' 4-bin profiles across positions:")
println("(cat eats fish | the dog runs fast)")
println()
for neuron in 1:5
    println("  Neuron $neuron:")
    for pos in 1:length(all_ids)
        t_start = (pos - 1) * 20 + 1
        t_end = pos * 20
        profile = SORNMemory.Bridge.normalize_rates_profile(result.spikes, t_start, t_end; n_bins=n_bins)
        bins = round.(profile[neuron, :], digits=3)
        word = SORNMemory.Tokenizer.decode(tok, [all_ids[pos]])[1]
        println("    $(rpad(word, 6)) bin=[$(join(bins, ", "))]")
    end
    println()
end

# Show SORN readout CAN decode (just not as well as FRP)
println("--- SORN readout vs FRP readout (same query) ---")
println()

for query in ["cat fish", "bird tree", "dog park"]
    ids = encode(tok, query)
    n_ids, n_scores = recall!(mem, ids; top_k=5, method=:neural)

    # Manually do SORN decode for comparison
    spikes = SORNMemory.Bridge.encode_tokens(mem.bridge, ids; timesteps_per_token=20, rng=MersenneTwister(mem.rng_seed))
    result = SORNMemory.SNN.Simulation.simulate!(mem.sorn, spikes; verbose=false)
    all_sims = zeros(size(mem.bridge.embedding_table, 1))
    for pos in 1:length(ids)
        t_start = (pos - 1) * 20 + 1
        t_end = pos * 20
        profile = SORNMemory.Bridge.normalize_rates_profile(result.spikes, t_start, t_end; n_bins=n_bins)
        rates = SORNMemory.Bridge.flatten_profile(profile[1:n_exc, :])
        _, _, sims = SORNMemory.Readout.decode_to_tokens(mem.readout, rates, mem.bridge.embedding_table; top_k=5)
        all_sims .+= sims
    end
    all_sims ./= length(ids)
    k = min(5, length(all_sims))
    sorn_indices = partialsortperm(all_sims, 1:k, rev=true)
    sorn_words = SORNMemory.Tokenizer.decode(tok, sorn_indices)
    frp_words = SORNMemory.Tokenizer.decode(tok, n_ids)

    println("Query: \"$query\" -> $(join(SORNMemory.Tokenizer.decode(tok, ids), ", "))")
    println("  SORN decode: $(join(sorn_words, ", "))")
    println("  FRP  decode: $(join(frp_words, ", "))")
    println()
end

# Show SORN weight evolution
println("--- SORN weight evolution ---")
println("Mean |W_EE|: $(round(mean(abs.(mem.sorn.W_EE.nzval)), digits=4)) (w_max=2.0)")
println("EE synapses: $(nnz(mem.sorn.W_EE))")
println("Frozen: $(mem.sorn.frozen)")
println()

println("--- Summary ---")
println("SORN produces structured temporal firing patterns per token.")
println("Each token drives a unique combination of neuron rates.")
println("The 4-bin temporal profiles capture firing evolution within a token.")
println()
println("SORN readout CAN decode tokens but with lower discriminability than FRP.")
println("STDP saturates weights toward w_max=2.0, collapsing attractor diversity.")
println()
println("SORN's value: biologically-plausible temporal dynamics.")
println("FRP's value: query-discriminative neural recall.")
