module Readout

using Random

export ReadoutLayer, decode_to_tokens, create_readout, train_readout!

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
    ra_norm = sqrt(sum(readout_activity .^ 2)) + 1e-10
    readout_activity ./= ra_norm

    similarities = embedding_table * readout_activity

    abs_sim = abs.(similarities)
    max_sim = maximum(abs_sim) + 1e-10
    scaled = similarities ./ (max_sim * readout.temperature)

    k = min(top_k, length(similarities))
    top_indices = partialsortperm(similarities, 1:k, rev=true)
    top_scores = similarities[top_indices]

    return top_indices, top_scores, similarities
end

function train_readout!(readout::ReadoutLayer, rates::AbstractVector{Float64},
                        embedding::AbstractVector{Float64}; lr::Float64=0.02, decay::Float64=0.003)
    y = readout.projection' * rates
    error = embedding - y
    readout.projection .+= lr * (rates * error' - decay * readout.projection)
end

end
