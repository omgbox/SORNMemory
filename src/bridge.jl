module Bridge

using Random
using SparseArrays

export TokenBridge, encode_tokens, decode_spikes, create_bridge

struct TokenBridge
    embedding_table::Matrix{Float64}
    token_masks::Vector{Vector{Int}}
    neurons_per_token::Int
    embed_dim::Int
    n_input::Int
    vocab_size::Int
    spike_rate::Float64
end

function create_bridge(; vocab_size::Int=1000, embed_dim::Int=32,
                       neurons_per_token::Int=16, spike_rate::Float64=0.3,
                       seed::Union{Int,Nothing}=nothing)
    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()

    embedding_table = randn(rng, vocab_size, embed_dim)
    for i in 1:vocab_size
        embedding_table[i, :] ./= sqrt(sum(x -> x^2, embedding_table[i, :])) + 1e-8
    end

    n_input = embed_dim * neurons_per_token

    token_masks = Vector{Vector{Int}}(undef, vocab_size)
    for i in 1:vocab_size
        n_active = max(1, round(Int, n_input * 0.1))
        perm = randperm(rng, n_input)
        token_masks[i] = sort(perm[1:n_active])
    end

    TokenBridge(embedding_table, token_masks, neurons_per_token, embed_dim, n_input,
                vocab_size, spike_rate)
end

function encode_tokens(bridge::TokenBridge, token_ids::Vector{Int};
                       timesteps_per_token::Int=20, dt::Float64=1.0)
    n_tokens = length(token_ids)
    total_timesteps = n_tokens * timesteps_per_token
    spikes = falses(bridge.n_input, total_timesteps)

    for (t_offset, token_id) in enumerate(token_ids)
        if token_id < 1 || token_id > bridge.vocab_size
            continue
        end

        active_neurons = bridge.token_masks[token_id]
        n_active = length(active_neurons)

        background_rate = bridge.spike_rate * 0.3
        token_rate = bridge.spike_rate

        t_start = (t_offset - 1) * timesteps_per_token + 1
        t_end = t_offset * timesteps_per_token

        for t in t_start:t_end
            for i in 1:bridge.n_input
                if i in active_neurons
                    if rand() < token_rate * dt
                        spikes[i, t] = true
                    end
                else
                    if rand() < background_rate * dt
                        spikes[i, t] = true
                    end
                end
            end
        end
    end

    return spikes
end

function decode_spikes(bridge::TokenBridge, spike_rates::AbstractVector{Float64})
    n_neurons_per_dim = bridge.neurons_per_token
    dim_rates = zeros(bridge.embed_dim)

    for d in 1:bridge.embed_dim
        start_idx = (d - 1) * n_neurons_per_dim + 1
        end_idx = d * n_neurons_per_dim
        if end_idx <= length(spike_rates)
            dim_rates[d] = sum(spike_rates[start_idx:end_idx]) / n_neurons_per_dim
        end
    end

    similarities = bridge.embedding_table * dim_rates
    return similarities
end

function normalize_rates(spike_matrix::AbstractMatrix{Bool}, window_start::Int, window_end::Int)
    n_neurons = size(spike_matrix, 1)
    rates = zeros(n_neurons)

    actual_end = min(window_end, size(spike_matrix, 2))
    window_size = max(1, actual_end - window_start + 1)

    for i in 1:n_neurons
        count = 0
        for t in window_start:actual_end
            if spike_matrix[i, t]
                count += 1
            end
        end
        rates[i] = count / window_size
    end

    return rates
end

function norm(v::AbstractVector)
    s = 0.0
    for x in v
        s += x * x
    end
    return sqrt(s)
end

end
