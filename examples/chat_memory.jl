using Pkg
Pkg.activate(".")

include(joinpath(@__DIR__, "..", "src", "SORNMemory.jl"))

using .SORNMemory
using .SORNMemory.Session: ChatSession, create_session, chat!, get_session_stats
using .SORNMemory.LLMInterface: test_connection, load_all_keys

println("=" ^ 60)
println("  SORN Memory — Chat with Episodic Memory")
println("=" ^ 60)
println()
println("Commands: 'quit', 'stats', 'history', 'quiet', 'verbose'")
println()

provider = select_provider(provider_name="nim")
println()

println("Testing connection...")
if !test_connection(provider)
    println("FAILED to connect. Check your API key and billing.")
    exit(1)
end

println()
println("Creating SORN memory network (300E + 75I = 375 neurons)...")
session = create_session(provider=provider, n_exc=300, vocab_size=1000, seed=42, verbose=true)

println()
println("Ready! Type your messages below.")
println("-" ^ 60)
println()

while true
    print("You: ")
    user_input = readline()
    user_input = strip(user_input)

    if isempty(user_input)
        continue
    end

    if user_input == "quit" || user_input == "exit"
        stats = get_session_stats(session)
        println()
        println("Session ended.")
        println("  Messages: $(stats.messages)")
        println("  SORN episodes: $(stats.memory.n_episodes)")
        println("  SORN synapses: $(stats.memory.n_synapses)")
        break
    end

    if user_input == "stats"
        stats = get_session_stats(session)
        println()
        println("Session Statistics:")
        println("  Provider: $(session.provider)")
        println("  Messages: $(stats.messages)")
        println("  Episodes stored: $(stats.memory.n_episodes)")
        println("  Total stored: $(stats.memory.total_stored)")
        println("  Network synapses: $(stats.memory.n_synapses)")
        println()
        continue
    end

    if user_input == "history"
        println()
        for msg in session.history
            println("  $msg")
        end
        println()
        continue
    end

    if user_input == "quiet"
        session.verbose = false
        println("  [verbose mode off]")
        println()
        continue
    end

    if user_input == "verbose"
        session.verbose = true
        println("  [verbose mode on]")
        println()
        continue
    end

    try
        response = chat!(session, user_input)
        println()
        println("Assistant: $response")
        println()
    catch e
        println()
        println("ERROR: $e")
        if occursin("401", string(e)) || occursin("Unauthorized", string(e))
            println("API key may be invalid. Check keys.txt.")
        elseif occursin("429", string(e)) || occursin("quota", string(e))
            println("Rate limited or out of quota. Wait or check billing.")
        elseif occursin("Timeout", string(e)) || occursin("connect", string(e))
            println("Network error. Check your internet connection.")
        end
        println()
    end
end
