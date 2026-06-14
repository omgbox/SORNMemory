module Utils

using Random

export poisson_sample, exponential_trace!

function poisson_sample(rate::Float64, rng::AbstractRNG)::Bool
    @inbounds return rand(rng) < rate
end

function exponential_trace!(trace::AbstractVector{Float64}, spikes::AbstractVector{Bool}, dt::Float64, tau::Float64)
    decay = exp(-dt / tau)
    @inbounds @simd for i in eachindex(trace)
        trace[i] = trace[i] * decay + Float64(spikes[i])
    end
end

end
