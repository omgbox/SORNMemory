module Readout

using Random

export ReadoutLayer, decode_to_tokens, create_readout

struct ReadoutLayer
    projection::Matrix{Float64}
    temperature::Float64
    embed_dim::Int
end

function create_readout(n_exc::Int, embed_dim::Int;
                        temperature::Float64=0.5,
                        seed::Union{Int,Nothing}=nothing)
    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()

    projection = randn(rng, n_exc, embed_dim)
    for i in 1:embed_dim
        projection[:, i] ./= sqrt(n_exc) + 1e-8
    end

    ReadoutLayer(projection, temperature, embed_dim)
end

function decode_to_tokens(readout::ReadoutLayer, spike_rates::AbstractVector{Float64},
                          embedding_table::Matrix{Float64}; top_k::Int=5)
    readout_activity = readout.projection' * spike_rates

    similarities = embedding_table * readout_activity

    abs_sim = abs.(similarities)
    max_sim = maximum(abs_sim) + 1e-10
    scaled = similarities ./ (max_sim * readout.temperature)

    k = min(top_k, length(similarities))
    top_indices = partialsortperm(similarities, 1:k, rev=true)
    top_scores = similarities[top_indices]

    return top_indices, top_scores, similarities
end

end
