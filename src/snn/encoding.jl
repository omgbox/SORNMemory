module Encoding

using Random

export poisson_encode, temporal_encode, generate_input

function poisson_encode(n_neurons::Int, n_timesteps::Int; rate::Float64=0.1,
                         dt::Float64=1.0, seed::Union{Int,Nothing}=nothing)
    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()
    spikes = falses(n_neurons, n_timesteps)
    p = rate * dt
    @inbounds for t in 1:n_timesteps, i in 1:n_neurons
        spikes[i, t] = rand(rng) < p
    end
    return spikes
end

function temporal_encode(intensities::AbstractVector{Float64}, n_timesteps::Int)
    n = length(intensities)
    spikes = falses(n, n_timesteps)
    @inbounds for i in 1:n
        t_spike = round(Int, (1.0 - intensities[i]) * n_timesteps)
        if 1 <= t_spike <= n_timesteps
            spikes[i, t_spike] = true
        end
    end
    return spikes
end

function generate_input(n_exc::Int, n_timesteps::Int; base_rate::Float64=0.1,
                         seed::Union{Int,Nothing}=nothing)
    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()
    spikes = falses(n_exc, n_timesteps)
    @inbounds for t in 1:n_timesteps, i in 1:n_exc
        spikes[i, t] = rand(rng) < base_rate
    end
    return spikes
end

end
