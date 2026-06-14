module EpisodicMemory

using Random
using SparseArrays

using ..SORNMemory: SNN
using ..SORNMemory.SNN.Network: SORN, create_sorn
using ..SORNMemory.SNN.Simulation: simulate!, SimResult
using ..SORNMemory.SNN.Encoding: poisson_encode
using ..Bridge: TokenBridge, create_bridge, encode_tokens, decode_spikes, normalize_rates
using ..Readout: ReadoutLayer, decode_to_tokens, create_readout

export EpisodicMemorySystem, store!, recall!, consolidate!, get_stats

mutable struct Episode
    token_ids::Vector{Int}
    timestamp::Float64
    n_timesteps::Int
    recall_count::Int
    importance::Float64
end

mutable struct EpisodicMemorySystem
    sorn::SORN
    bridge::TokenBridge
    readout::ReadoutLayer
    episodes::Vector{Episode}
    max_episodes::Int
    timesteps_per_token::Int
    consolidation_interval::Int
    store_count::Int
end

function create_episodic_memory(; n_exc::Int=300, n_inh::Int=75,
                                 vocab_size::Int=1000, embed_dim::Int=32,
                                 neurons_per_token::Int=16,
                                 max_episodes::Int=500,
                                 timesteps_per_token::Int=20,
                                 exc_w::Float64=1.0,
                                 seed::Union{Int,Nothing}=nothing)
    bridge = create_bridge(vocab_size=vocab_size, embed_dim=embed_dim,
                           neurons_per_token=neurons_per_token, seed=seed)

    n_input = bridge.n_input
    net = create_sorn(n_exc=n_exc, n_inh=n_inh, n_input=n_input,
                      connectivity=0.15, exc_w=exc_w, seed=seed)
    net.W_in.nzval .*= 0.18

    readout = create_readout(n_exc, embed_dim, seed=seed)

    EpisodicMemorySystem(
        net, bridge, readout,
        Episode[],
        max_episodes,
        timesteps_per_token,
        50,
        0
    )
end

function store!(mem::EpisodicMemorySystem, token_ids::Vector{Int})
    if isempty(token_ids)
        return nothing
    end

    input_spikes = encode_tokens(mem.bridge, token_ids,
                                 timesteps_per_token=mem.timesteps_per_token)

    result = simulate!(mem.sorn, input_spikes; verbose=false)

    timestamp = time()
    episode = Episode(token_ids, timestamp, size(input_spikes, 2), 0, 1.0)
    push!(mem.episodes, episode)

    mem.store_count += 1

    if length(mem.episodes) > mem.max_episodes
        consolidate!(mem)
    end

    return nothing
end

function jaccard_similarity(a::Vector{Int}, b::Vector{Int})
    set_a = Set(a)
    set_b = Set(b)
    overlap = length(intersect(set_a, set_b))
    union_sz = length(union(set_a, set_b))
    return union_sz > 0 ? overlap / union_sz : 0.0
end

function subsequence_bonus(query::Vector{Int}, episode::Vector{Int})
    bonus = 1.0
    for window in 2:min(3, length(query), length(episode))
        for i_q in 1:(length(query) - window + 1)
            q_seq = query[i_q:i_q+window-1]
            for i_e in 1:(length(episode) - window + 1)
                if q_seq == episode[i_e:i_e+window-1]
                    bonus *= 1.5
                end
            end
        end
    end
    return bonus
end

function nearest_episode(episodes::Vector{Episode}, query::Vector{Int})
    best_idx = 0
    best_score = -Inf
    best_tokens = Int[]

    for (i, ep) in enumerate(episodes)
        jac = jaccard_similarity(query, ep.token_ids)
        seq_bonus = subsequence_bonus(query, ep.token_ids)
        score = jac * seq_bonus
        if score > best_score
            best_score = score
            best_idx = i
            best_tokens = ep.token_ids
        end
    end

    return best_tokens, best_score, best_idx
end

function recall!(mem::EpisodicMemorySystem, query_token_ids::Vector{Int};
                 top_k::Int=5, n_sim_timesteps::Int=100,
                 method::Symbol=:episode)
    if isempty(query_token_ids)
        return Int[], Float64[]
    end

    if method == :episode
        best_tokens, best_score, best_idx = nearest_episode(mem.episodes, query_token_ids)

        if best_idx > 0
            ep = mem.episodes[best_idx]
            ep.recall_count += 1
            ep.importance += 0.1
        end

        if isempty(best_tokens)
            return Int[], Float64[]
        end

        k = min(top_k, length(best_tokens))
        scores = fill(best_score, k)
        return best_tokens[1:k], scores
    end

    input_spikes = encode_tokens(mem.bridge, query_token_ids,
                                 timesteps_per_token=mem.timesteps_per_token)

    n_timesteps = min(n_sim_timesteps, size(input_spikes, 2))
    query_input = input_spikes[:, 1:n_timesteps]

    result = simulate!(mem.sorn, query_input; verbose=false)

    n_exc = mem.sorn.exc.n
    window_start = max(1, n_timesteps - 20)
    spike_rates = normalize_rates(result.spikes, window_start, n_timesteps)

    readout_rates = spike_rates[1:n_exc]

    indices, scores, _ = decode_to_tokens(mem.readout, readout_rates,
                                          mem.bridge.embedding_table,
                                          top_k=top_k)

    for ep in mem.episodes
        overlap = length(intersect(Set(ep.token_ids), Set(indices)))
        if overlap > 0
            ep.recall_count += 1
            ep.importance += 0.1
        end
    end

    return indices, scores
end

function consolidate!(mem::EpisodicMemorySystem)
    if isempty(mem.episodes)
        return nothing
    end

    sort!(mem.episodes, by = e -> e.importance / (e.recall_count + 1), rev=true)

    n_keep = min(mem.max_episodes, length(mem.episodes))
    if n_keep < length(mem.episodes)
        resize!(mem.episodes, n_keep)
    end

    for ep in mem.episodes
        ep.importance *= 0.99
    end

    return nothing
end

function get_stats(mem::EpisodicMemorySystem)
    n_episodes = length(mem.episodes)
    total_stored = mem.store_count
    avg_importance = n_episodes > 0 ?
        sum(e.importance for e in mem.episodes) / n_episodes : 0.0
    avg_recalls = n_episodes > 0 ?
        sum(e.recall_count for e in mem.episodes) / n_episodes : 0.0

    n_exc = mem.sorn.exc.n
    n_inh = mem.sorn.inh.n
    n_synapses = nnz(mem.sorn.W_EE)

    return (
        n_episodes = n_episodes,
        total_stored = total_stored,
        avg_importance = avg_importance,
        avg_recalls = avg_recalls,
        n_exc = n_exc,
        n_inh = n_inh,
        n_synapses = n_synapses
    )
end

end
