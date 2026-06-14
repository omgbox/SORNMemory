SORNMemory — SORN-powered episodic memory for LLMs. Demo project.

A self-organizing spiking neural network (SORN) that stores chat history as temporal spike patterns and retrieves relevant context for LLM prompts.

## Setup

```bash
git clone https://github.com/omgbox/SORNMemory.git
cd SORNMemory
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## API Key

Create `keys.txt` in the project root:

```
nvida Nim key: nvapi-your-key-here
```

Only NVIDIA NIM is supported.

## Run

**No API key needed:**
```bash
julia --project=. examples/verify_sorn.jl
julia --project=. examples/pattern_learning.jl
```

**With API key:**
```bash
julia --project=. examples/chat_memory.jl
```

Commands inside chat: `quit`, `stats`, `history`, `quiet`, `verbose`

## How It Works

1. Each message is tokenized (hash → token ID)
2. Token IDs drive a 375-neuron SORN with 5 plasticity rules (STDP, ISTDP, synaptic scaling, intrinsic plasticity, structural plasticity)
3. On recall, the current message tokens are matched against stored episodes by Jaccard similarity
4. The best-matching episode tokens are formatted as context and injected into the LLM prompt
5. The LLM response is also stored back into SORN as a new episode

## Requirements

- Julia 1.10+
- HTTP.jl, JSON3.jl
- NVIDIA NIM API key

## Project Structure

```
SORNMemory/
├── src/
│   ├── SORNMemory.jl          # Main module
│   ├── bridge.jl              # Token ↔ spike encoding
│   ├── readout.jl             # Spike pattern → token decoding
│   ├── episodic_memory.jl     # store!/recall!/consolidate! — core memory API
│   ├── llm_interface.jl       # NVIDIA NIM provider
│   ├── context_injection.jl   # Memory → prompt formatting
│   ├── session.jl             # Chat loop orchestrator
│   └── snn/                   # Bundled SORN neural network source
├── examples/
│   ├── verify_sorn.jl         # SORN pipeline verification (no API key)
│   ├── pattern_learning.jl    # SORN learning demo (no API key)
│   └── chat_memory.jl         # Full chat with SORN memory
├── keys.template.txt          # API key format template
├── LICENSE                    # MIT License
├── Project.toml
└── README.md
```

## References

- Lazar et al. (2009) "SORN: a self-organizing recurrent neural network"
- Turrigiano & Nelson (2000) "Homeostatic plasticity in developing networks"
