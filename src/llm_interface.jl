module LLMInterface

using HTTP
using JSON3

export NIMProvider, Message, CompletionResult
export complete, create_nim_provider
export load_all_keys, test_connection

struct Message
    role::Symbol
    content::String
end

function Base.show(io::IO, m::Message)
    content_preview = length(m.content) > 60 ? m.content[1:60] * "..." : m.content
    print(io, "[", m.role, "] ", content_preview)
end

struct CompletionResult
    content::String
    token_usage::Int
    model::String
    provider::String
end

function Base.show(io::IO, r::CompletionResult)
    print(io, CompletionResult, "(", r.provider, "/", r.model, ", ", r.token_usage, " tokens)")
end

abstract type LLMProvider end

mutable struct NIMProvider <: LLMProvider
    api_key::String
    model::String
    base_url::String
    temperature::Float64
    max_tokens::Int
end

function Base.show(io::IO, p::NIMProvider)
    masked = mask_key(p.api_key)
    print(io, "NIM(", p.model, ", ", masked, ")")
end

function mask_key(key::String)
    if length(key) <= 12
        return key[1:4] * "..." * key[end-3:end]
    end
    return key[1:8] * "..." * key[end-3:end]
end

# --- Key Loading ---

function load_all_keys(; path::String="")::Dict{String,String}
    if isempty(path)
        candidates = [
            joinpath(@__DIR__, "..", "keys.txt"),
            joinpath(@__DIR__, "..", "..", "keys.txt"),
        ]
        path = ""
        for c in candidates
            if isfile(c)
                path = c
                break
            end
        end
    end

    if isempty(path) || !isfile(path)
        return Dict{String,String}()
    end

    keys = Dict{String,String}()
    content = strip(read(path, String))

    nim_count = 0

    for line in split(content, '\n')
        line = strip(line)
        isempty(line) && continue

        if startswith(line, "nvida Nim key:") || startswith(line, "nvidia nim key:") || startswith(line, "nvida NIM key:")
            key = strip(split(line, ":")[2])
            nim_count += 1
            keys["nim_$nim_count"] = key
            if nim_count == 1
                keys["nim"] = key
            end
        end
    end

    return keys
end

function load_api_key(provider::String; path::String="")::String
    keys = load_all_keys(path=path)
    if haskey(keys, provider)
        return keys[provider]
    end
    error("No API key found for provider '$provider' in keys.txt")
end

# --- Provider Constructors ---

function create_nim_provider(; api_key::String="",
                              model::String="meta/llama-3.1-8b-instruct",
                              base_url::String="https://integrate.api.nvidia.com/v1",
                              temperature::Float64=0.7,
                              max_tokens::Int=1024)
    if isempty(api_key)
        api_key = load_api_key("nim")
    end
    NIMProvider(api_key, model, base_url, temperature, max_tokens)
end

# --- NVIDIA NIM Provider ---

function complete(provider::NIMProvider, messages::Vector{Message};
                  temperature::Union{Float64,Nothing}=nothing,
                  max_tokens::Union{Int,Nothing}=nothing)
    temp = something(temperature, provider.temperature)
    tokens = something(max_tokens, provider.max_tokens)

    api_messages = [
        Dict("role" => string(m.role), "content" => m.content)
        for m in messages
    ]

    body = Dict(
        "model" => provider.model,
        "messages" => api_messages,
        "temperature" => temp,
        "max_tokens" => tokens
    )

    url = "$(provider.base_url)/chat/completions"

    response = HTTP.post(
        url,
        [
            "Content-Type" => "application/json",
            "Authorization" => "Bearer $(provider.api_key)"
        ],
        JSON3.write(body);
        retry=false,
        connect_timeout=30,
        request_timeout=60
    )

    if response.status != 200
        error("NVIDIA NIM API error $(response.status): $(String(response.body))")
    end

    result = JSON3.read(String(response.body))
    content = result.choices[1].message.content
    usage = get(result, :usage, nothing)
    total_tokens = usage !== nothing ? get(usage, :total_tokens, 0) : 0

    return CompletionResult(content, total_tokens, provider.model, "nim")
end

# --- Connection Testing ---

function test_connection(provider::NIMProvider)::Bool
    println("  Testing NVIDIA NIM connection...")
    println("  Endpoint: $(provider.base_url)/chat/completions")
    println("  Model: $(provider.model)")
    println("  Key: $(mask_key(provider.api_key))")

    try
        body = JSON3.write(Dict(
            "model" => provider.model,
            "messages" => [Dict("role" => "user", "content" => "Say hello in one word.")],
            "max_tokens" => 10
        ))

        response = HTTP.post(
            "$(provider.base_url)/chat/completions",
            ["Content-Type" => "application/json", "Authorization" => "Bearer $(provider.api_key)"],
            body;
            retry=false, connect_timeout=10, request_timeout=30, status_exception=false
        )

        if response.status == 200
            result = JSON3.read(String(response.body))
            reply = strip(result.choices[1].message.content)
            println("  OK! Response: \"$(reply)\"")
            return true
        else
            data = JSON3.read(String(response.body))
            msg = get(get(data, :error, Dict()), :message, "Unknown error")
            println("  FAILED ($(response.status)): $msg")
            return false
        end
    catch e
        println("  ERROR: $e")
        return false
    end
end

function test_connection(provider::LLMProvider)::Bool
    println("  Unknown provider type: $(typeof(provider))")
    return false
end

end
