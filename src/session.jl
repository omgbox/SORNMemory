module Session

using ..EpisodicMemory: EpisodicMemorySystem, create_episodic_memory, store!, recall!, consolidate!, get_stats
using ..LLMInterface: LLMProvider, NIMProvider, Message, CompletionResult, complete, create_nim_provider, test_connection, load_all_keys
using ..Tokenizer: Tokenizer, encode, decode, build_vocab!
using ..ContextInjection: format_memory_context, inject_memory_context

export ChatSession, create_session, chat!, get_session_stats, select_provider

mutable struct ChatSession
    memory::EpisodicMemorySystem
    provider::LLMProvider
    tokenizer::Tokenizer
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
    if length(s.memory.episodes) > 0
        println(io, "  Recent episode: ", replay_sorn_tokens(s))
    end
end

function select_provider(; provider_name::String="", api_key::String="")::LLMProvider
    if isempty(api_key)
        api_key = get(load_all_keys(), "nim", "")
    end
    println("  Using provider: nim")
    return create_nim_provider(api_key=api_key)
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

    tokenizer = Tokenizer(vocab_size)
    build_vocab!(tokenizer, [
        "hello hi hey goodbye bye thanks thank please sorry yes no ok okay",
        "the a an is are was were be been being have has had do does did done",
        "i you he she it we they me him her us them my your his its our their",
        "this that these those some any all each every both few many much more most",
        "in on at to for with by from of about into through during before after",
        "and but or nor so yet because if when while until unless though although",
        "say tell ask answer know think believe feel want need see hear look find",
        "good bad big small hot cold long short old new high low near far right wrong",
        "one two three four five six seven eight nine ten hundred thousand million billion first last next",
        "what where when why who how which whose whom",
        "can will would should could may might must shall",
        "here there now then always never often sometimes",
        "up down in out on off over under above below",
        "people place thing time world city country state",
        "work life home day week month year name number way part",
        "great large little important different same other many",
        "come go get make take give use find keep start",
        "like just also very really well even still too",
        "only own old new such much more most",
        "paris france europe america china japan england london",
        "russia germany italy spain india brazil australia",
        "capital city country continent ocean river mountain lake",
        "money economy business market trade bank price cost value tax debt",
        "gdp growth inflation deflation recession depression crisis",
        "government policy law right power state nation political",
        "war peace army force attack defense nuclear weapon",
        "health medical disease virus vaccine drug treatment",
        "technology computer internet data ai robot software hardware",
        "science physics chemistry biology math research theory experiment",
        "education school university student teacher class course degree",
        "music art film book story write read culture history language",
        "food water air energy sun moon star planet earth nature",
        "family friend love care support hope faith trust truth",
        "question answer problem solution idea plan change move step",
        "help need want have get give take make do say go come",
    ])

    history = Message[Message(:system, system_prompt)]

    session = ChatSession(
        memory, provider, tokenizer, history, system_prompt,
        max_context_tokens, recall_top_k, n_sim_timesteps,
        0, verbose
    )

    return session
end

function simple_tokenize(session::ChatSession, text::AbstractString)
    return encode(session.tokenizer, text)
end

function replay_sorn_tokens(session::ChatSession)
    mem = session.memory
    n_exc = mem.sorn.exc.n
    if length(mem.episodes) == 0
        return "no episodes stored yet"
    end

    last = mem.episodes[end]
    tids = last.token_ids
    words = decode(session.tokenizer, tids)
    if length(tids) <= 3
        return "[$(join(tids, ", "))] -> [$(join(words, ", "))]"
    end
    return "[$(join(tids[1:3], ", ")), ...] ($(length(tids)) tokens) -> [$(join(words[1:3], ", ")), ...]"
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

    user_tokens = simple_tokenize(session, user_message)
    user_words = decode(session.tokenizer, user_tokens)

    if session.verbose
        println("  [SORN] Tokens: $(user_tokens) -> [$(join(user_words, ", "))]")
    end

    recalled_indices, recalled_scores = recall!(session.memory, user_tokens;
                                                top_k=session.recall_top_k,
                                                n_sim_timesteps=session.n_sim_timesteps,
                                                method=:episode)

    store!(session.memory, user_tokens)

    if session.verbose
        if !isempty(recalled_indices)
            recalled_words = decode(session.tokenizer, recalled_indices)
            scored = sort(collect(zip(recalled_indices, recalled_words, recalled_scores)), by=x->x[3], rev=true)
            println("  [SORN] Recalled $(length(recalled_indices)) tokens:")
            for (idx, word, score) in scored
                if abs(score) > 0.1
                    println("         $idx ($word)  relevance: $(round(score, digits=3))")
                end
            end
        else
            println("  [SORN] No relevant context recalled")
        end
    end

    context_text = format_memory_context(recalled_indices, recalled_scores;
                                         vocab_reverse=session.tokenizer.id_to_word)

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
        println("  [SORN] Episodes: $(stats.n_episodes), Synapses: $(stats.n_synapses), Last episode: $(replay_sorn_tokens(session))")
    end

    result = complete(session.provider, msgs)

    push!(session.history, Message(:user, user_message))
    push!(session.history, Message(:assistant, result.content))

    if length(session.history) > 50
        system_msg = session.history[1]
        session.history = session.history[end-30:end]
        insert!(session.history, 1, system_msg)
    end

    response_tokens = simple_tokenize(session, result.content)
    if session.verbose
        rwords = decode(session.tokenizer, response_tokens[1:min(5, end)])
        println("  [SORN] Storing response: $(response_tokens[1:min(5, end)]) -> [$(join(rwords, ", "))]... ($(length(response_tokens)) tokens)")
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
