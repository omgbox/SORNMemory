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

## Benefits

- **Beyond context windows** — LLM context limits are fixed (8K-128K tokens). SORN stores unlimited conversation history as compressed spike-weight patterns, not raw text. No sliding window, no truncation.
- **Content-addressable recall** — Retrieves past episodes by topical overlap (Jaccard similarity on token IDs), not recency. A question about something said 500 messages ago triggers the same recall as something said 5 messages ago.
- **Inference-time cost efficiency** — Instead of prepending the entire conversation history (growing token costs linearly with each turn), SORN injects only ~5-10 relevant token IDs. Cheaper and faster per call.
- **No pretrained embeddings** — No dependency on BERT, SentenceTransformers, or any external embedding model. The SORN network learns temporal patterns purely through spike-timing-dependent plasticity.
- **Online learning** — Every message updates the network immediately. No batch training, no finetuning, no gradient descent.

## Use Cases

- **Persistent chatbots** — Give an LLM long-term memory beyond its context window without ballooning API costs.
- **Research on biological memory** — Study how STDP, synaptic scaling, and intrinsic plasticity interact to store and recall sequences.
- **Privacy-sensitive applications** — All memory is in local SORN weight matrices, not in a cloud vector database.
- **Low-resource memory** — SORN runs on CPU (375 neurons, 13K synapses). No GPU needed.
- **Explainable memory retrieval** — Every recall shows exactly which token overlap triggered the match, unlike opaque vector embedding similarity.

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
