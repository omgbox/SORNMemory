module Network

using Random
using SparseArrays
using ..Neurons: ExcitatoryPool, InhibitoryPool, step_pool!
using ..Plasticity: apply_pair_stdp!, apply_inhibitory_stdp!, apply_synaptic_scaling!,
                    apply_structural_plasticity!
using ..Utils: exponential_trace!

export SORN, create_sorn, freeze!, unfreeze!

mutable struct SORN
    exc::ExcitatoryPool
    inh::InhibitoryPool

    W_EE::SparseMatrixCSC{Float64,Int}
    W_EI::SparseMatrixCSC{Float64,Int}
    W_IE::SparseMatrixCSC{Float64,Int}
    W_in::SparseMatrixCSC{Float64,Int}

    exc_trace::Vector{Float64}
    inh_trace::Vector{Float64}
    exc_rates::Vector{Float64}

    I_rec::Vector{Float64}
    I_input::Vector{Float64}

    dt::Float64
    n_input::Int
    frozen::Bool
end

function create_sorn(; n_exc::Int=80, n_inh::Int=20, n_input::Int=100,
                      connectivity::Float64=0.15, exc_w::Float64=1.0,
                      inh_w::Float64=-5.0, dt::Float64=1.0,
                      seed::Union{Int,Nothing}=nothing)

    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()

    exc = ExcitatoryPool(n_exc)
    inh = InhibitoryPool(n_inh)

    W_EE = spzeros(n_exc, n_exc)
    W_EI = spzeros(n_exc, n_inh)
    W_IE = spzeros(n_inh, n_exc)
    W_in = spzeros(n_exc, n_input)

    for j in 1:n_exc, i in 1:n_exc
        i != j && rand(rng) < connectivity && (W_EE[i, j] = exc_w * (0.5 + rand(rng)))
    end

    for j in 1:n_inh, i in 1:n_exc
        rand(rng) < connectivity && (W_EI[i, j] = abs(inh_w) * rand(rng))
    end

    for j in 1:n_exc, i in 1:n_inh
        rand(rng) < connectivity && (W_IE[i, j] = abs(inh_w) * rand(rng))
    end

    for j in 1:n_input, i in 1:n_exc
        rand(rng) < connectivity && (W_in[i, j] = exc_w * (0.5 + rand(rng)))
    end

    n_total = n_exc + n_inh

    SORN(
        exc, inh,
        W_EE, W_EI, W_IE, W_in,
        zeros(n_exc), zeros(n_inh), zeros(n_total),
        zeros(n_total), zeros(n_exc),
        dt, n_input, false
    )
end

function compute_currents!(net::SORN, input_spikes::AbstractVector{Bool})
    n_exc, n_inh = net.exc.n, net.inh.n

    fill!(net.I_rec, 0.0)
    fill!(net.I_input, 0.0)

    @inbounds for j in 1:n_exc
        if net.exc.fired[j]
            for idx in nzrange(net.W_EE, j)
                net.I_rec[rowvals(net.W_EE)[idx]] += net.W_EE.nzval[idx]
            end
            for idx in nzrange(net.W_IE, j)
                net.I_rec[n_exc + rowvals(net.W_IE)[idx]] += net.W_IE.nzval[idx]
            end
        end
    end

    @inbounds for j in 1:n_inh
        if net.inh.fired[j]
            for idx in nzrange(net.W_EI, j)
                net.I_rec[rowvals(net.W_EI)[idx]] -= net.W_EI.nzval[idx]
            end
        end
    end

    @inbounds for j in 1:net.n_input
        if input_spikes[j]
            for idx in nzrange(net.W_in, j)
                net.I_input[rowvals(net.W_in)[idx]] += net.W_in.nzval[idx]
            end
        end
    end
end

function freeze!(net::SORN)
    net.frozen = true
end

function unfreeze!(net::SORN)
    net.frozen = false
end

end
