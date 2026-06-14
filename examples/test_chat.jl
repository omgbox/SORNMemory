using SORNMemory

# Create session first (so tokenizer and memory are in sync)
session = create_session(verbose=true)

# Store episodes through the session's tokenizer
episodes = [
    "the cat eats fish in the kitchen",
    "a dog runs fast through the park",
    "i ate a big red apple for lunch",
    "blue bird flies high in the sky",
    "the horse runs fast on the farm",
]
for text in episodes
    ids = encode(session.tokenizer, text)
    store!(session.memory, ids)
    println("Stored: $text")
end
train_readout!(session.memory; alpha=1.0)

println()
println("--- Chat test ---")
println()

result = chat!(session, "what animals did i talk about?")
println("User: what animals did i talk about?")
println("Assistant: $result")
println()

result2 = chat!(session, "what did the cat eat?")
println("User: what did the cat eat?")
println("Assistant: $result2")
