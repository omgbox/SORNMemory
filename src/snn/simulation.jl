module Simulation

using Random
using SparseArrays
using ..Network: SORN, compute_currents!
using ..Neurons: step_pool!
using ..Plasticity: apply_pair_stdp!, apply_inhibitory_stdp!, apply_synaptic_scaling!,
                    apply_structural_plasticity!
using ..Utils: exponential_trace!

export simulate!, SimResult

mutable struct SimResult
    spikes::Matrix{Bool}
    voltages::Matrix{Float64}
    weights::Vector{Float64}
    n_timesteps::Int
    dt::Float64
end

function simulate!(net::SORN, input::AbstractMatrix{Bool}; dt::Float64=net.dt,
                   verbose::Bool=false, scale_every::Int=200,
                   struct_every::Int=500, record_every::Int=100)
    n_timesteps = size(input, 2)
    n_exc = net.exc.n
    n_total = n_exc + net.inh.n

    spikes = falses(n_total, n_timesteps)
    voltages = zeros(n_total, n_timesteps)
    weights = Float64[]

    target_rate = 0.05
    current_rates = zeros(n_total)
    rate_window = 200.0

    if verbose
        println("Starting simulation: $n_timesteps timesteps")
    end

    for t in 1:n_timesteps
        input_col = @view input[:, t]

        compute_currents!(net, input_col)

        exc_current = @view net.I_rec[1:n_exc]
        inh_current = @view net.I_rec[(n_exc+1):end]
        total_exc = net.I_input .+ exc_current

        step_pool!(net.exc, total_exc, dt)
        step_pool!(net.inh, inh_current, dt)

        @inbounds @simd for i in 1:n_exc
            spikes[i, t] = net.exc.fired[i]
            voltages[i, t] = net.exc.v[i]
        end
        @inbounds @simd for i in 1:net.inh.n
            spikes[n_exc + i, t] = net.inh.fired[i]
            voltages[n_exc + i, t] = net.inh.v[i]
        end

        exc_fired = @view net.exc.fired[1:n_exc]
        all_fired = @view spikes[:, t]
        exponential_trace!(net.exc_trace, exc_fired, dt, 20.0)
        exponential_trace!(net.inh_trace, all_fired, dt, 20.0)

        apply_pair_stdp!(net.W_EE, net.exc_trace, exc_fired, exc_fired, dt)
        apply_inhibitory_stdp!(net.W_EI, net.inh_trace, exc_fired, net.inh.fired)

        @inbounds for i in 1:n_total
            current_rates[i] += (Float64(spikes[i, t]) - current_rates[i]) * dt / rate_window
        end

        if t % scale_every == 0
            apply_synaptic_scaling!(net.W_EE, target_rate, current_rates[1:n_exc], dt)
        end

        if t % struct_every == 0
            net.W_EE = apply_structural_plasticity!(net.W_EE)
        end

        if t % record_every == 0
            push!(weights, nnz(net.W_EE) > 0 ? sum(net.W_EE) / nnz(net.W_EE) : 0.0)
        end

        if verbose && t % 500 == 0
            rate = sum(all_fired) / n_total
            println("  t=$t  rate=$(round(rate * 100, digits=1))%")
        end
    end

    if verbose
        total = sum(spikes)
        avg = total / (n_total * n_timesteps)
        println("Done. Total spikes: $total  avg rate: $(round(avg * 100, digits=2))%")
    end

    SimResult(spikes, voltages, weights, n_timesteps, dt)
end

end
