module LLMInterface

using HTTP
using JSON3

export OpenAIProvider, GeminiProvider, NIMProvider, CerebrasProvider, Message, CompletionResult
export complete, create_openai_provider, create_gemini_provider, create_nim_provider, create_cerebras_provider
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

mutable struct OpenAIProvider <: LLMProvider
    api_key::String
    model::String
    base_url::String
    temperature::Float64
    max_tokens::Int
end

function Base.show(io::IO, p::OpenAIProvider)
    masked = mask_key(p.api_key)
    print(io, "OpenAI(", p.model, ", ", masked, ")")
end

mutable struct GeminiProvider <: LLMProvider
    api_key::String
    model::String
    temperature::Float64
    max_tokens::Int
end

function Base.show(io::IO, p::GeminiProvider)
    masked = mask_key(p.api_key)
    print(io, "Gemini(", p.model, ", ", masked, ")")
end

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

mutable struct CerebrasProvider <: LLMProvider
    api_key::String
    model::String
    base_url::String
    temperature::Float64
    max_tokens::Int
end

function Base.show(io::IO, p::CerebrasProvider)
    masked = mask_key(p.api_key)
    print(io, "Cerebras(", p.model, ", ", masked, ")")
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
        path = joinpath(@__DIR__, "..", "..", "keys.txt")
    end

    if !isfile(path)
        return Dict{String,String}()
    end

    keys = Dict{String,String}()
    content = strip(read(path, String))

    openai_count = 0
    gemini_count = 0
    nim_count = 0
    cerebras_count = 0

    for line in split(content, '\n')
        line = strip(line)
        isempty(line) && continue

        if startswith(line, "openai key:")
            key = strip(split(line, "openai key:")[2])
            openai_count += 1
            keys["openai_$openai_count"] = key
            if openai_count == 1
                keys["openai"] = key
            end
        elseif startswith(line, "gemini key:")
            key = strip(split(line, "gemini key:")[2])
            gemini_count += 1
            keys["gemini_$gemini_count"] = key
            if gemini_count == 1
                keys["gemini"] = key
            end
        elseif startswith(line, "nvida Nim key:") || startswith(line, "nvidia nim key:") || startswith(line, "nvida NIM key:")
            key = strip(split(line, ":")[2])
            nim_count += 1
            keys["nim_$nim_count"] = key
            if nim_count == 1
                keys["nim"] = key
            end
        elseif startswith(line, "cerebras ai key:") || startswith(line, "cerebras key:")
            key = strip(split(line, ":")[2])
            cerebras_count += 1
            keys["cerebras_$cerebras_count"] = key
            if cerebras_count == 1
                keys["cerebras"] = key
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

function create_openai_provider(; api_key::String="",
                                 model::String="gpt-4o-mini",
                                 base_url::String="https://api.openai.com/v1",
                                 temperature::Float64=0.7,
                                 max_tokens::Int=1024)
    if isempty(api_key)
        api_key = load_api_key("openai")
    end
    OpenAIProvider(api_key, model, base_url, temperature, max_tokens)
end

function create_gemini_provider(; api_key::String="",
                                 model::String="gemini-2.0-flash",
                                 temperature::Float64=0.7,
                                 max_tokens::Int=1024)
    if isempty(api_key)
        api_key = load_api_key("gemini")
    end
    GeminiProvider(api_key, model, temperature, max_tokens)
end

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

function create_cerebras_provider(; api_key::String="",
                                   model::String="gpt-oss-120b",
                                   base_url::String="https://api.cerebras.ai/v1",
                                   temperature::Float64=0.7,
                                   max_tokens::Int=1024)
    if isempty(api_key)
        api_key = load_api_key("cerebras")
    end
    CerebrasProvider(api_key, model, base_url, temperature, max_tokens)
end

# --- OpenAI Provider ---

function complete(provider::OpenAIProvider, messages::Vector{Message};
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
        error("OpenAI API error $(response.status): $(String(response.body))")
    end

    result = JSON3.read(String(response.body))
    content = result.choices[1].message.content
    usage = get(result, :usage, nothing)
    total_tokens = usage !== nothing ? get(usage, :total_tokens, 0) : 0

    return CompletionResult(content, total_tokens, provider.model, "openai")
end

# --- Gemini Provider ---

function convert_to_gemini_contents(messages::Vector{Message})
    contents = Dict{String,Any}[]
    system_instruction = nothing

    for msg in messages
        if msg.role == :system
            system_instruction = Dict("parts" => [Dict("text" => msg.content)])
        else
            role = msg.role == :assistant ? "model" : "user"
            push!(contents, Dict(
                "role" => role,
                "parts" => [Dict("text" => msg.content)]
            ))
        end
    end

    return contents, system_instruction
end

function complete(provider::GeminiProvider, messages::Vector{Message};
                  temperature::Union{Float64,Nothing}=nothing,
                  max_tokens::Union{Int,Nothing}=nothing)
    temp = something(temperature, provider.temperature)
    tokens = something(max_tokens, provider.max_tokens)

    contents, system_instruction = convert_to_gemini_contents(messages)

    body = Dict{String,Any}(
        "contents" => contents,
        "generationConfig" => Dict(
            "temperature" => temp,
            "maxOutputTokens" => tokens
        )
    )

    if system_instruction !== nothing
        body["systemInstruction"] = system_instruction
    end

    url = "https://generativelanguage.googleapis.com/v1beta/models/$(provider.model):generateContent"

    response = HTTP.post(
        url,
        [
            "Content-Type" => "application/json",
            "x-goog-api-key" => provider.api_key
        ],
        JSON3.write(body);
        retry=false,
        connect_timeout=30,
        request_timeout=60
    )

    if response.status != 200
        error("Gemini API error $(response.status): $(String(response.body))")
    end

    result = JSON3.read(String(response.body))

    candidates = get(result, :candidates, nothing)
    if candidates === nothing || isempty(candidates)
        error("Gemini returned no candidates: $(String(response.body))")
    end

    candidate = candidates[1]
    content_parts = candidate.content.parts
    text = content_parts[1].text

    usage = get(result, :usageMetadata, nothing)
    total_tokens = usage !== nothing ? get(usage, :totalTokenCount, 0) : 0

    return CompletionResult(text, total_tokens, provider.model, "gemini")
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

# --- Cerebras Provider ---

function complete(provider::CerebrasProvider, messages::Vector{Message};
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
        "max_completion_tokens" => tokens
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
        request_timeout=120
    )

    if response.status != 200
        error("Cerebras API error $(response.status): $(String(response.body))")
    end

    result = JSON3.read(String(response.body))
    msg = result.choices[1].message

    content = ""
    if haskey(msg, :content) && msg.content !== nothing
        content = msg.content
    elseif haskey(msg, :reasoning) && msg.reasoning !== nothing
        content = msg.reasoning
    end

    usage = get(result, :usage, nothing)
    total_tokens = usage !== nothing ? get(usage, :total_tokens, 0) : 0

    return CompletionResult(content, total_tokens, provider.model, "cerebras")
end

# --- Connection Testing ---

function test_connection(provider::OpenAIProvider)::Bool
    println("  Testing OpenAI connection...")
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

function test_connection(provider::GeminiProvider)::Bool
    println("  Testing Gemini connection...")
    println("  Endpoint: generativelanguage.googleapis.com")
    println("  Model: $(provider.model)")
    println("  Key: $(mask_key(provider.api_key))")

    try
        body = JSON3.write(Dict(
            "contents" => [Dict("parts" => [Dict("text" => "Say hello in one word.")])],
            "generationConfig" => Dict("maxOutputTokens" => 10)
        ))

        url = "https://generativelanguage.googleapis.com/v1beta/models/$(provider.model):generateContent"
        response = HTTP.post(
            url,
            ["Content-Type" => "application/json", "x-goog-api-key" => provider.api_key],
            body;
            retry=false, connect_timeout=10, request_timeout=30, status_exception=false
        )

        if response.status == 200
            result = JSON3.read(String(response.body))
            reply = strip(result.candidates[1].content.parts[1].text)
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

function test_connection(provider::CerebrasProvider)::Bool
    println("  Testing Cerebras connection...")
    println("  Endpoint: $(provider.base_url)/chat/completions")
    println("  Model: $(provider.model)")
    println("  Key: $(mask_key(provider.api_key))")

    try
        body = JSON3.write(Dict(
            "model" => provider.model,
            "messages" => [Dict("role" => "user", "content" => "Say hello in one word.")],
            "max_completion_tokens" => 100
        ))

        response = HTTP.post(
            "$(provider.base_url)/chat/completions",
            ["Content-Type" => "application/json", "Authorization" => "Bearer $(provider.api_key)"],
            body;
            retry=false, connect_timeout=10, request_timeout=60, status_exception=false
        )

        if response.status == 200
            result = JSON3.read(String(response.body))
            msg = result.choices[1].message
            reply = ""
            if haskey(msg, :content) && msg.content !== nothing
                reply = strip(msg.content)
            elseif haskey(msg, :reasoning) && msg.reasoning !== nothing
                reply = strip(msg.reasoning)
            end
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
