module SNN

include("utils.jl")
include("neurons.jl")
include("plasticity.jl")
include("network.jl")
include("simulation.jl")
include("encoding.jl")

using .Utils
using .Neurons
using .Plasticity
using .Network
using .Simulation
using .Encoding

end
