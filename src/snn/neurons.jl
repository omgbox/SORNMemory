module Neurons

export ExcitatoryPool, InhibitoryPool, step_pool!

mutable struct ExcitatoryPool
    n::Int
    v::Vector{Float64}
    v_rest::Float64
    v_reset::Float64
    v_thresh::Vector{Float64}
    v_thresh_rest::Float64
    tau_m::Float64
    r_m::Float64
    tau_adapt::Float64
    alpha::Float64
    fired::Vector{Bool}
end

function ExcitatoryPool(n::Int; v_rest=-70.0, v_reset=-75.0, v_thresh=-55.0,
                         tau_m=20.0, r_m=15.0, tau_adapt=100.0, alpha=0.5)
    ExcitatoryPool(
        n, fill(v_rest, n), v_rest, v_reset,
        fill(v_thresh, n), v_thresh,
        tau_m, r_m, tau_adapt, alpha,
        falses(n)
    )
end

function reset!(pool::ExcitatoryPool, idx::Int)
    @inbounds begin
        pool.v[idx] = pool.v_reset
        pool.fired[idx] = false
    end
end

mutable struct InhibitoryPool
    n::Int
    v::Vector{Float64}
    u::Vector{Float64}
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    fired::Vector{Bool}
end

function InhibitoryPool(n::Int; a=0.1, b=0.2, c=-65.0, d=8.0)
    InhibitoryPool(n, fill(-65.0, n), fill(-14.0, n), a, b, c, d, falses(n))
end

@inline function step_lif!(v::Float64, I::Float64, dt::Float64, tau_m::Float64, r_m::Float64)::Float64
    exp_decay = exp(-dt / tau_m)
    return -70.0 + (v - (-70.0)) * exp_decay + r_m * I * (1.0 - exp_decay)
end

@inline function step_izhikevich!(v::Float64, u::Float64, I::Float64, a::Float64, b::Float64)::Tuple{Float64,Float64}
    dv = 0.04v * v + 5.0v + 140.0 - u + I
    du = a * (b * v - u)
    return v + dv, u + du
end

function step_pool!(pool::ExcitatoryPool, current::AbstractVector{Float64}, dt::Float64)
    thresh_decay = 1.0 - exp(-dt / pool.tau_adapt)
    @inbounds @simd for i in 1:pool.n
        pool.v[i] = step_lif!(pool.v[i], current[i], dt, pool.tau_m, pool.r_m)
        pool.fired[i] = pool.v[i] >= pool.v_thresh[i]
        if pool.fired[i]
            pool.v[i] = pool.v_reset
            pool.v_thresh[i] += pool.alpha * (pool.v_thresh[i] - pool.v_thresh_rest)
        else
            pool.v_thresh[i] += (pool.v_thresh_rest - pool.v_thresh[i]) * thresh_decay
        end
    end
end

function step_pool!(pool::InhibitoryPool, current::AbstractVector{Float64}, dt::Float64)
    @inbounds @simd for i in 1:pool.n
        pool.v[i], pool.u[i] = step_izhikevich!(pool.v[i], pool.u[i], current[i], pool.a, pool.b)
        pool.fired[i] = pool.v[i] >= 30.0
        if pool.fired[i]
            pool.v[i] = pool.c
            pool.u[i] += pool.d
        end
    end
end

end
