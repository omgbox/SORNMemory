# SORN Memory

A Julia library that uses a Self-Organizing Recurrent Neural Network (SORN) as episodic memory for LLMs.

## What This Does

SORN Memory gives LLMs an online-learning episodic memory that never needs retraining. While the LLM stays frozen, SORN learns conversation patterns through 5 biologically plausible plasticity mechanisms — no backpropagation.

## Setup

```bash
# 1. Install Julia 1.10+
# 2. Create keys.txt in the research/ directory (one level up from this project):
echo "nvida Nim key: nvapi-your-key-here" > ../keys.txt

# 3. Run without API key (tests SORN + encoding only):
julia --project=. examples/pattern_learning.jl

# 4. Run with your provider:
julia --project=. examples/chat_memory.jl
```

## keys.txt Format

The file `../keys.txt` (relative to this project root) supports 4 providers:

```
openai key: sk-...
gemini key: AIza...
nvida Nim key: nvapi-...
cerebras ai key: csk-...
```

Add only the providers you use. The chat example auto-detects available keys and prompts you to pick one.

## Usage: Chat with SORN Memory

The simplest way to use it is the interactive chat:

```bash
julia --project=. examples/chat_memory.jl
```

Select a provider when prompted, then talk normally. SORN stores each message, learns temporal patterns, and injects recalled context into the LLM prompt.

**Commands during chat:**
- `stats` — show memory usage and SORN network stats
- `history` — show conversation history
- `switch` — switch LLM provider mid-session
- `quit` — exit

## Usage: SORN Memory Without LLM

You can use the SORN memory system standalone (no API key needed):

```julia
include("src/SORNMemory.jl")
using .SORNMemory

# Create memory system (375 neurons, 1000 token vocab)
mem = create_episodic_memory(n_exc=300, vocab_size=1000, seed=42)

# Store a sequence of token IDs
store!(mem, [1, 2, 3, 4, 5])

# Query with a partial sequence — SORN recalls related tokens
indices, scores = recall!(mem, [1]; top_k=5)

println(indices)   # [53, 100, 93, 14, 73] — recalled token IDs
println(scores)    # [0.79, 0.73, 0.72, 0.69, 0.66] — relevance scores

# Check memory statistics
stats = get_stats(mem)
println(stats.total_stored)  # number of episodes stored
println(stats.n_synapses)    # SORN network synapses
```

## Usage: Programmatic Chat (Pick Your Provider)

```julia
include("src/SORNMemory.jl")
using .SORNMemory

# Auto-pick provider from keys.txt
session = create_session()

# Or specify a provider directly:
session = create_session(provider_name="nim")

# Chat
response = chat!(session, "Where is Paris?")
println(response)  # "Paris is the capital of France..."

# Stats
stats = get_session_stats(session)
println(stats.messages)             # 1
println(stats.memory.n_episodes)    # episodes stored in SORN
```

You can also create a provider manually:

```julia
using .SORNMemory.LLMInterface: create_nim_provider

nim = create_nim_provider(api_key="nvapi-...", model="meta/llama-3.1-70b-instruct")
session = create_session(provider=nim)
```

## Available Providers

| Provider | Default Model | Parameter | Notes |
|----------|---------------|-----------|-------|
| OpenAI | `gpt-4o-mini` | `temperature`, `max_tokens` | Standard format |
| Gemini | `gemini-2.0-flash` | `temperature`, `max_tokens` | Different API format |
| NVIDIA NIM | `meta/llama-3.1-8b-instruct` | `temperature`, `max_tokens` | OpenAI-compatible |
| Cerebras | `gpt-oss-120b` | `temperature`, `max_tokens` | Use `max_completion_tokens >= 100` for reasoning models |

## SORN Parameters

Edit these when creating memory for different behavior:

```julia
mem = create_episodic_memory(
    n_exc=300,        # excitatory neurons (default)
    n_inh=75,         # inhibitory neurons (20% of n_exc)
    vocab_size=1000,  # max distinct tokens
    embed_dim=32,     # embedding dimensions per token
    timesteps_per_token=20,  # simulation steps per token
    seed=42           # reproducible results
)
```

## Requirements

- Julia 1.10+
- HTTP.jl, JSON3.jl (installed automatically by `Pkg.instantiate()`)
- At least one API key in `../keys.txt`

## Project Structure

```
SORNMemory/
├── src/
│   ├── SORNMemory.jl          # Main module
│   ├── bridge.jl              # Token ↔ spike encoding
│   ├── readout.jl             # Spike pattern → token decoding
│   ├── episodic_memory.jl     # store!/recall!/consolidate! — core memory API
│   ├── llm_interface.jl       # 4 LLM providers (OpenAI, Gemini, NIM, Cerebras)
│   ├── context_injection.jl   # Memory → prompt formatting
│   ├── session.jl             # Chat loop orchestrator
│   └── snn/                   # Bundled SORN neural network source
├── examples/
│   ├── verify_sorn.jl         # SORN pipeline verification (no API key)
│   ├── pattern_learning.jl    # SORN learning demo (no API key)
│   └── chat_memory.jl         # Full chat with SORN memory
├── AGENTS.md                  # Developer guide for AI agents
├── keys.template.txt          # API key format template
├── LICENSE                    # MIT License
├── Project.toml
└── README.md
```

## References

- Lazar et al. (2009) "SORN: a self-organizing recurrent neural network"
- Turrigiano & Nelson (2000) "Homeostatic plasticity in developing networks"
