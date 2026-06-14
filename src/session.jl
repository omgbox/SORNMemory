module Session

using ..EpisodicMemory: EpisodicMemorySystem, create_episodic_memory, store!, recall!, consolidate!, get_stats
using ..LLMInterface: LLMProvider, OpenAIProvider, GeminiProvider, NIMProvider, CerebrasProvider, Message, CompletionResult, complete, create_openai_provider, create_gemini_provider, create_nim_provider, create_cerebras_provider, test_connection, load_all_keys
using ..ContextInjection: format_memory_context, inject_memory_context

export ChatSession, create_session, chat!, get_session_stats, select_provider

mutable struct ChatSession
    memory::EpisodicMemorySystem
    provider::LLMProvider
    history::Vector{Message}
    system_prompt::String
    max_context_tokens::Int
    recall_top_k::Int
    n_sim_timesteps::Int
    message_count::Int
    verbose::Bool
end

function Base.show(io::IO, s::ChatSession)
    println(io, "ChatSession:")
    println(io, "  Provider: ", s.provider)
    println(io, "  Messages: ", s.message_count)
    println(io, "  Memory episodes: ", length(s.memory.episodes))
    println(io, "  SORN neurons: ", s.memory.sorn.exc.n, "E + ", s.memory.sorn.inh.n, "I")
end

function select_provider(; provider_name::String="", api_key::String="")::LLMProvider
    keys = load_all_keys()

    if !isempty(provider_name)
        name = lowercase(provider_name)
    else
        available = String[]
        haskey(keys, "openai") && push!(available, "openai")
        haskey(keys, "gemini") && push!(available, "gemini")
        haskey(keys, "nim") && push!(available, "nim")
        haskey(keys, "cerebras") && push!(available, "cerebras")

        if isempty(available)
            error("No API keys found in keys.txt")
        end

        if length(available) == 1
            name = available[1]
            println("  Only provider available: $name")
        else
            println()
            println("  Available providers:")
            for (i, p) in enumerate(available)
                println("    $i) $p")
            end
            print("  Select provider [1]: ")
            choice = readline()
            choice = strip(choice)
            if isempty(choice)
                choice = "1"
            end
            idx = parse(Int, choice)
            name = available[idx]
        end
    end

    println("  Using provider: $name")

    if name == "openai"
        key = !isempty(api_key) ? api_key : get(keys, "openai", "")
        return create_openai_provider(api_key=key)
    elseif name == "gemini"
        key = !isempty(api_key) ? api_key : get(keys, "gemini", "")
        return create_gemini_provider(api_key=key)
    elseif name == "nim"
        key = !isempty(api_key) ? api_key : get(keys, "nim", "")
        return create_nim_provider(api_key=key)
    elseif name == "cerebras"
        key = !isempty(api_key) ? api_key : get(keys, "cerebras", "")
        return create_cerebras_provider(api_key=key)
    else
        error("Unknown provider: $name. Use 'openai', 'gemini', 'nim', or 'cerebras'.")
    end
end

function create_session(; provider::Union{LLMProvider,Nothing}=nothing,
                         provider_name::String="",
                         memory::Union{EpisodicMemorySystem,Nothing}=nothing,
                         system_prompt::String="You are a helpful assistant with episodic memory powered by a spiking neural network.",
                         max_context_tokens::Int=200,
                         recall_top_k::Int=5,
                         n_sim_timesteps::Int=100,
                         n_exc::Int=300,
                         vocab_size::Int=1000,
                         exc_w::Float64=1.0,
                         verbose::Bool=true,
                         seed::Union{Int,Nothing}=nothing)
    if provider === nothing
        provider = select_provider(provider_name=provider_name)
    end

    if memory === nothing
        memory = create_episodic_memory(n_exc=n_exc, vocab_size=vocab_size, exc_w=exc_w, seed=seed)
    end

    history = Message[Message(:system, system_prompt)]

    session = ChatSession(
        memory, provider, history, system_prompt,
        max_context_tokens, recall_top_k, n_sim_timesteps,
        0, verbose
    )

    return session
end

function simple_tokenize(text::AbstractString; vocab_size::Int=1000)
    words = split(lowercase(text), r"[^a-z0-9]+", keepempty=false)
    token_ids = Int[]
    for word in words
        h = hash(word) % vocab_size + 1
        push!(token_ids, h)
    end
    return token_ids
end

function replay_sorn_tokens(mem::EpisodicMemorySystem)
    n_exc = mem.sorn.exc.n
    if length(mem.episodes) == 0
        return "no episodes stored yet"
    end

    last = mem.episodes[end]
    tids = last.token_ids
    if length(tids) <= 3
        return "[$(join(tids, ", "))]"
    end
    return "[$(join(tids[1:3], ", ")), ...] ($(length(tids)) tokens)"
end

function chat!(session::ChatSession, user_message::AbstractString)::String
    session.message_count += 1

    user_tokens = simple_tokenize(user_message)

    if session.verbose
        println("  [SORN] Storing tokens: $(user_tokens)")
    end
    store!(session.memory, user_tokens)

    recalled_indices, recalled_scores = recall!(session.memory, user_tokens;
                                                top_k=session.recall_top_k,
                                                n_sim_timesteps=session.n_sim_timesteps,
                                                method=:episode)

    if session.verbose
        if !isempty(recalled_indices)
            scored = sort(collect(zip(recalled_indices, recalled_scores)), by=x->x[2], rev=true)
            println("  [SORN] Recalled $(length(recalled_indices)) tokens:")
            for (idx, score) in scored
                if abs(score) > 0.1
                    println("         token $idx  (relevance: $(round(score, digits=3)))")
                end
            end
        else
            println("  [SORN] No relevant context recalled")
        end
    end

    context_text = format_memory_context(recalled_indices, recalled_scores)

    msgs = inject_memory_context(session.history, context_text; position=:before_user)

    push!(msgs, Message(:user, user_message))

    if session.verbose
        if !isempty(context_text)
            println("  [SORN] Injected context into prompt:")
            for line in split(context_text, '\n')
                println("         $line")
            end
        end
        println("  [LLM] Sending prompt with $(length(msgs)) messages to $(typeof(session.provider).name.name)...")
        stats = get_stats(session.memory)
        println("  [SORN] Episodes: $(stats.n_episodes), Synapses: $(stats.n_synapses), Last episode: $(replay_sorn_tokens(session.memory))")
    end

    result = complete(session.provider, msgs)

    push!(session.history, Message(:user, user_message))
    push!(session.history, Message(:assistant, result.content))

    if length(session.history) > 50
        system_msg = session.history[1]
        session.history = session.history[end-30:end]
        insert!(session.history, 1, system_msg)
    end

    response_tokens = simple_tokenize(result.content)
    if session.verbose
        println("  [SORN] Storing response: $(response_tokens[1:min(5, end)])... ($(length(response_tokens)) tokens)")
    end
    store!(session.memory, response_tokens)

    if session.message_count % 10 == 0
        consolidate!(session.memory)
        if session.verbose
            println("  [SORN] Consolidation cycle completed")
        end
    end

    return result.content
end

function get_session_stats(session::ChatSession)
    mem_stats = get_stats(session.memory)
    return (
        messages = session.message_count,
        memory = mem_stats
    )
end

end
