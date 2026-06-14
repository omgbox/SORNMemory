module FRP

using Random

export FRPState, create_frp, encode_frp

struct FRPState
    W_in::Matrix{Float64}
    bias::Vector{Float64}
    n_input::Int
    n_reservoir::Int
end

function create_frp(; n_input::Int=512, n_reservoir::Int=300, seed::Union{Int,Nothing}=nothing)
    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()
    W_in = randn(rng, n_input, n_reservoir) ./ sqrt(n_input)
    bias = zeros(n_reservoir)
    FRPState(W_in, bias, n_input, n_reservoir)
end

function encode_frp(frp::FRPState, rates::AbstractVector{Float64})
    return tanh.(frp.W_in' * rates .+ frp.bias)
end

end
