module Tokenizer

export Tokenizer, encode, decode, build_vocab!

mutable struct Tokenizer
    word_to_id::Dict{String,Int}
    id_to_word::Dict{Int,String}
    vocab_size::Int
    unk_id::Int
    reserved::Int

    function Tokenizer(vocab_size::Int=4096)
        t = new(Dict{String,Int}(), Dict{Int,String}(), vocab_size, 0, 2)
        t.word_to_id["<UNK>"] = 1
        t.id_to_word[1] = "<UNK>"
        t.unk_id = 1
        t.word_to_id["<PAD>"] = 2
        t.id_to_word[2] = "<PAD>"
        return t
    end
end

function build_vocab!(tokenizer::Tokenizer, texts::Vector{String})
    word_counts = Dict{String,Int}()
    for text in texts
        for word in split(lowercase(text), r"[^a-z0-9]+"; keepempty=false)
            word_counts[word] = get(word_counts, word, 0) + 1
        end
    end

    sorted = sort(collect(word_counts); by=x -> x[2], rev=true)
    max_new = tokenizer.vocab_size - tokenizer.reserved

    for (word, _) in sorted[1:min(end, max_new)]
        if !haskey(tokenizer.word_to_id, word)
            id = length(tokenizer.word_to_id) + 1
            tokenizer.word_to_id[word] = id
            tokenizer.id_to_word[id] = word
        end
    end
end

function encode(tokenizer::Tokenizer, text::AbstractString)::Vector{Int}
    words = split(lowercase(text), r"[^a-z0-9]+"; keepempty=false)
    ids = Int[]
    for word in words
        if haskey(tokenizer.word_to_id, word)
            push!(ids, tokenizer.word_to_id[word])
        elseif length(tokenizer.word_to_id) < tokenizer.vocab_size
            id = length(tokenizer.word_to_id) + 1
            tokenizer.word_to_id[word] = id
            tokenizer.id_to_word[id] = word
            push!(ids, id)
        else
            push!(ids, tokenizer.unk_id)
        end
    end
    return ids
end

function decode(tokenizer::Tokenizer, ids::AbstractVector{<:Integer})::Vector{String}
    return [get(tokenizer.id_to_word, id, "<UNK>") for id in ids]
end

end
