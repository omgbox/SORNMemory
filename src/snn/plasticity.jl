module Plasticity

using Random
using SparseArrays

export apply_pair_stdp!, apply_inhibitory_stdp!, apply_synaptic_scaling!,
       apply_structural_plasticity!

function apply_pair_stdp!(W::SparseMatrixCSC{Float64,Int}, pre_trace::AbstractVector{Float64},
                          post_fired::AbstractVector{Bool}, pre_fired::AbstractVector{Bool},
                          dt::Float64; w_max::Float64=2.0)
    a_plus = 0.0005
    a_minus = 0.0007
    rv = rowvals(W)
    nz = nonzeros(W)

    @inbounds for j in 1:size(W, 2)
        if pre_fired[j]
            pt_j = pre_trace[j]
            for idx in nzrange(W, j)
                i = rv[idx]
                if post_fired[i]
                    nz[idx] += a_minus * pt_j
                    nz[idx] += a_plus * (1.0 - nz[idx] / w_max) * pt_j
                    nz[idx] = clamp(nz[idx], 0.0, w_max)
                end
            end
        end
    end
end

function apply_inhibitory_stdp!(W::SparseMatrixCSC{Float64,Int}, pre_trace::AbstractVector{Float64},
                                 post_fired::AbstractVector{Bool}, pre_fired::AbstractVector{Bool})
    a_minus = 0.005
    rv = rowvals(W)
    nz = nonzeros(W)

    @inbounds for j in 1:size(W, 2)
        if pre_fired[j]
            pt_j = pre_trace[j]
            for idx in nzrange(W, j)
                i = rv[idx]
                if post_fired[i]
                    nz[idx] -= a_minus * pt_j
                    nz[idx] = max(nz[idx], 0.0)
                end
            end
        end
    end
end

function apply_synaptic_scaling!(W::SparseMatrixCSC{Float64,Int}, target_rate::Float64,
                                 current_rates::AbstractVector{Float64}, dt::Float64)
    tau_scale = 500.0
    scale_rate = dt / tau_scale
    nz = nonzeros(W)

    @inbounds for i in 1:size(W, 1)
        if current_rates[i] > 0.0
            factor = exp(scale_rate * (target_rate - current_rates[i]))
            for idx in nzrange(W, i)
                nz[idx] *= factor
                nz[idx] = clamp(nz[idx], 0.0, 2.0)
            end
        end
    end
end

function apply_structural_plasticity!(W::SparseMatrixCSC{Float64,Int}; p_form::Float64=0.001, p_elim::Float64=0.005)
    n_post, n_pre = size(W)
    rv = rowvals(W)
    nz = nonzeros(W)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    @inbounds for j in 1:n_pre
        for idx in nzrange(W, j)
            if rand() >= p_elim
                push!(rows, rv[idx])
                push!(cols, j)
                push!(vals, nz[idx])
            end
        end
    end

    target_nnz = round(Int, n_post * n_pre * 0.15)
    new_needed = target_nnz - length(rows)

    if new_needed > 0
        existing = Set{Tuple{Int,Int}}()
        sizehint!(existing, length(rows) + new_needed)
        for k in eachindex(rows)
            push!(existing, (rows[k], cols[k]))
        end

        added = 0
        max_attempts = new_needed * 5
        attempts = 0
        while added < new_needed && attempts < max_attempts
            r = rand(1:n_post)
            c = rand(1:n_pre)
            if (r, c) ∉ existing
                push!(rows, r)
                push!(cols, c)
                push!(vals, 0.1 * rand())
                push!(existing, (r, c))
                added += 1
            end
            attempts += 1
        end
    end

    return sparse(rows, cols, vals, n_post, n_pre)
end

end
