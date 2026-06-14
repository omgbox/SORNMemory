using SORNMemory
using Statistics: mean

tok = Tokenizer.Tokenizer()
mem = create_episodic_memory(seed=42)

episodes = [
    "the cat eats fish in the kitchen",
    "a dog runs fast through the park",
    "i ate a big red apple for lunch",
    "a blue bird flies high in the sky",
    "the horse runs fast on the farm",
    "a small fish swims in the blue sea",
    "the white cat sleeps on the red sofa",
    "green apples grow on the tree",
    "the dog eats the big apple from the ground",
    "a black bird sings in the green tree",
]
for text in episodes
    ids = encode(tok, text)
    store!(mem, ids)
    println("Stored: $text")
end
train_readout!(mem; alpha=1.0)
println()

function show_query(tok, mem, desc, query)
    ids = encode(tok, query)
    ep_ids, ep_scores = recall!(mem, ids; top_k=5, method=:episode)
    n_ids, n_scores = recall!(mem, ids; top_k=5, method=:neural)
    println("Query: \"$query\"  ->  ", join(decode(tok, ids), ", "))
    println("  Episode: ", join(decode(tok, ep_ids), " "))
    println("  FRP:     ", join(decode(tok, n_ids), ", "))
    println()
end

show_query(tok, mem, "cat fish", "cat fish")
show_query(tok, mem, "bird tree", "bird tree")
show_query(tok, mem, "dog park apple", "dog park apple")
show_query(tok, mem, "horse sea", "horse sea")
show_query(tok, mem, "kitchen apple", "kitchen apple")
show_query(tok, mem, "green bird", "green bird")
