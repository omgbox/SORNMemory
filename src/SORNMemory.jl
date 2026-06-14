module SORNMemory

function clear_compiled_cache()
    vstr = string(VERSION.major, ".", VERSION.minor)
    cache_root = joinpath(homedir(), ".julia", "compiled", vstr)
    targets = ["SORNMemory", "SNN"]
    for target in targets
        cached = joinpath(cache_root, target)
        if ispath(cached)
            try
                rm(cached; recursive=true, force=true)
            catch
            end
        end
    end
end

clear_compiled_cache()

include("snn/SNN.jl")
using .SNN
using .SNN.Network: SORN, create_sorn, freeze!, unfreeze!
using .SNN.Simulation: simulate!, SimResult
using .SNN.Encoding: poisson_encode

include("tokenizer.jl")
include("bridge.jl")
include("readout.jl")
include("frp_memory.jl")
include("episodic_memory.jl")
include("llm_interface.jl")
include("context_injection.jl")
include("session.jl")

using .Tokenizer
using .Bridge
using .Readout
using .FRP
using .EpisodicMemory
using .LLMInterface
using .ContextInjection
using .Session

export clear_compiled_cache, Tokenizer, encode, decode, build_vocab!
export TokenBridge, create_bridge, encode_tokens, decode_spikes, normalize_rates_profile, flatten_profile
export ReadoutLayer, create_readout, decode_to_tokens, train_readout!
export FRPState, create_frp, encode_frp
export EpisodicMemorySystem, create_episodic_memory, store!, recall!, consolidate!, get_stats, train_readout!, freeze!, unfreeze!
export LLMProvider, NIMProvider
export Message, CompletionResult, complete
export create_nim_provider
export load_all_keys, load_api_key, test_connection
export format_memory_context, inject_memory_context
export ChatSession, create_session, chat!, get_session_stats, select_provider

end
