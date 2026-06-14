module ContextInjection

using ..LLMInterface: Message

export inject_memory_context, format_memory_context, build_memory_message

function format_memory_context(token_ids::AbstractVector{<:Integer}, scores::AbstractVector{<:AbstractFloat};
                               vocab_reverse::Union{Dict{Int,String},Nothing}=nothing)
    if isempty(token_ids)
        return ""
    end

    lines = String[]
    for (id, score) in zip(token_ids, scores)
        if score > 0.1
            if vocab_reverse !== nothing && haskey(vocab_reverse, id)
                token_text = vocab_reverse[id]
            else
                token_text = "token_$(id)"
            end
            push!(lines, "  - \"$(token_text)\" (relevance: $(round(score, digits=3)))")
        end
    end

    if isempty(lines)
        return ""
    end

    context = "[SORN Memory Context]\nRelevant prior context:\n" * join(lines, "\n")
    return context
end

function build_memory_message(context::String; position::Symbol=:before_user)
    if isempty(context)
        return nothing
    end

    content = "$(context)\n\nUse this context if relevant to the current conversation. If not relevant, ignore it and respond normally."

    return Message(:system, content)
end

function inject_memory_context(messages::Vector{Message}, context::String;
                               position::Symbol=:before_user)
    if isempty(context)
        return copy(messages)
    end

    memory_msg = build_memory_message(context)
    if memory_msg === nothing
        return copy(messages)
    end

    result = Message[]

    if position == :before_user
        for msg in messages
            if msg.role == :user && isempty(result)
                push!(result, memory_msg)
            end
            push!(result, msg)
        end
    elseif position == :after_system
        pushed_memory = false
        for msg in messages
            push!(result, msg)
            if msg.role == :system && !pushed_memory
                push!(result, memory_msg)
                pushed_memory = true
            end
        end
        if !pushed_memory
            insert!(result, 1, memory_msg)
        end
    else
        push!(result, memory_msg)
        append!(result, messages)
    end

    return result
end

end
