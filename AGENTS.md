# AGENTS.md

## What This Is

SORNMemory — demo project using a Self-Organizing Recurrent Neural Network (SORN) as episodic memory for LLMs. SORN learns temporal patterns via 5 plasticity mechanisms without backpropagation.

## Run

```bash
# No API key needed — step-by-step SORN memory pipeline verification
julia --project=. examples/verify_sorn.jl

# No API key needed — verifies SORN + encoding pipeline
julia --project=. examples/pattern_learning.jl

# Chat with SORN memory (verbose by default, shows [SORN] activity)
julia --project=. examples/chat_memory.jl
```

## Providers

Single provider: NVIDIA NIM. API key loaded from `keys.txt`:

```
nvida Nim key: nvapi-...
```

| Provider | Endpoint | Default Model |
|----------|----------|---------------|
| NVIDIA NIM | `integrate.api.nvidia.com/v1` | `meta/llama-3.1-8b-instruct` |

## Architecture

SORN source is bundled at `src/snn/SNN.jl` — included via relative path (`include("snn/SNN.jl")` from SORNMemory.jl). Not a Julia package dependency.

Module load order (must stay this order):
1. `tokenizer.jl` — bidirectional word-level tokenizer with online vocab growth
2. `bridge.jl` — token ↔ spike encoding (one-hot-like patterns, 30% spike rate for active neurons)
2. `readout.jl` — spike rates → token scores (trained readout, Widrow-Hoff delta rule + ridge regression)
3. `episodic_memory.jl` — `store!`/`recall!`/`consolidate!` wrapping SORN
4. `llm_interface.jl` — NVIDIA NIM provider via `complete()` interface
5. `context_injection.jl` — formats recalled tokens into prompt context
6. `session.jl` — orchestrates chat loop with provider selection

## Key Data Flow

```
Text → Tokenizer.encode (word → ID, online vocab growth)
     → Bridge.encode_tokens (Poisson spike train, 512 input neurons)
     → SORN.simulate! (300E + 75I, STDP + 4 other plasticity rules)
     → normalize_rates (spike matrix → firing rates)
     → Readout.decode_to_tokens (cosine similarity with embedding table, normalized)
     → Tokenizer.decode (ID → word, reverse vocab lookup)
     → ContextInjection (format as system message with actual words)
     → provider.complete() (NIM)
```

Readout training happens inline in `store!` (per-token delta rule) and can be refreshed via `train_readout!(mem)` (ridge regression, re-simulates all episodes with current SORN state).

## Debugging Gotchas (Hard-Earned)

### `SubString{String}` from Piped stdin

On Windows, when stdin is a pipe (PowerShell `|`), `readline()` returns `SubString{String}` (zero-copy view), NOT `String`. Julia's dispatch is strict — `SubString{String}` does not match a `::String` signature. Use `::AbstractString` for all functions accepting string input:

### Use Abstract Type Signatures for Cross-Module Functions

When a function takes arguments that traverse module boundaries (especially from `include`d submodules), use abstract container types. Concrete types like `Vector{Int}` may not dispatch correctly across module scopes — the caller might produce `Vector{Int64}` which is a different type in Julia's dispatch system:

```julia
# DO:
function format_memory_context(token_ids::AbstractVector{<:Integer}, ...)

# DON'T:
function format_memory_context(token_ids::Vector{Int}, ...)
```

This applies to all exported functions that receive data from other modules.

**Critical: `readline()` returns `SubString{String}` when piped.** On Windows, when stdin is a pipe (PowerShell `|`), `readline()` returns `SubString{String}` (zero-copy view), NOT `String`. Julia's dispatch is strict — `SubString{String}` does not match a `::String` signature. All functions accepting string input must use `::AbstractString`:

```julia
# DO:
function chat!(session::ChatSession, user_message::AbstractString)::String

# DON'T:
function chat!(session::ChatSession, user_message::String)::String
```

This is why NIM appeared to produce a `MethodError` — it was never a provider issue. `SubString{String}` from piped stdin hit the `::String` type barrier.

### HTTP.jl v2.x Deprecated readtimeout

HTTP.jl v2 renamed `readtimeout` to `request_timeout`. Using `readtimeout` emits a deprecation warning. All 3 occur in `llm_interface.jl` (in each `complete()` and `test_connection()` method). Fix: replace `readtimeout=` with `request_timeout=`.

## SORN Parameters

- 300 excitatory + 75 inhibitory = 375 neurons
- 15% sparse connectivity (`SparseMatrixCSC`)
- Input weights: `exc_w=3.0` (needed for sparse token encoding to trigger spikes)
- Spike rate: 30% for active token neurons, 9% background
- Network fires at ~10-12% rate with token input

## Critical: Weight Matrix Orientation

Columns = source, rows = target (counterintuitive):
- `W_EE[i,j]` = weight from E neuron j → E neuron i
- `W_EI[i,j]` = weight from I neuron j → E neuron i (inhibition, subtracted)
- `W_in[i,j]` = weight from input neuron j → E neuron i

## Julia 1.12 Caveat

Windows Julia 1.12 LLVM JIT crashes with `EXCEPTION_ACCESS_VIOLATION` if `@threads` is used on complex loops. All inner loops use `@simd` only. Never add `@threads` to neuron/synapse loops.

## Sparse Matrix Iteration

All loops use CSC iteration: `nzrange(W, j)`, `rowvals(W)`, `nonzeros(W)`. Never use `W[i,j]` in hot loops — O(log nnz) per access.

## Recall Methods

`recall!` accepts `method::Symbol`:

- **`:episode`** (default) — nearest-neighbor search over stored episodes by TF-IDF weighted Jaccard similarity + subsequence matching. Returns tokens from best-matching episode. `n_sim_timesteps` is ignored. **This is the primary recall path used by `chat!`** in `session.jl`.
- **`:neural`** — trained readout via `decode_to_tokens`. During `store!`, per-token Widrow-Hoff delta rule trains the readout to map SORN firing rates → token embeddings. Call `train_readout!(mem)` for ridge regression batch training (re-simulates all episodes with current SORN state, one-shot optimal solution). Output is normalized to prevent score blowup. **Discriminative power is limited** by SORN attractor dynamics (same-length queries produce similar firing patterns).

Example:
```julia
recall!(mem, [42, 17]; top_k=5, method=:episode)
recall!(mem, [42, 17]; top_k=5, method=:neural)  # trained, not random
train_readout!(mem; alpha=1.0)  # ridge regression batch refresh
```

The `:episode` method is used in `session.jl`'s `chat!` loop — recalled tokens are injected as context into the LLM prompt.

## Critical: STDP Porting Bug (Root Cause of Weight Collapse)

The sparse CSC port of `apply_pair_stdp!` had a **sign error**. The original dense version uses purely additive STDP (both terms potentiate coincident firing):

```julia
# ORIGINAL (dense, correct):
W[i,j] += a_minus * trace       # potentiation at coincident firing
W[i,j] += a_plus * (1 - W/w_max) * trace  # weight-dependent potentiation
clamp(W[i,j], 0, w_max)

# BUGGY sparse port:
nz[idx] -= a_minus * trace       # ← WRONG: should be +=
nz[idx] += a_plus * (1 - w) * trace  # ← WRONG: should be (1 - w/w_max)
```

Consequences of the bug:
- Net-depressive STDP drives ALL weights to 0 whenever firing rate is high
- The `clamp(nz, 0, 1)` immediately truncates weights > 1.0, making the drop faster
- Combined with `exc_w=3.0` and 512 input neurons, E rate hits 99% within 50 timesteps

**Fix applied:**
1. `-= a_minus` → `+= a_minus`
2. `(1 - nz)` → `(1 - nz / w_max)` with `w_max=2.0`
3. Clamp `[0, w_max]` instead of `[0, 1.0]`

## SORNMemory Input Drive vs Original SORN

The original SORN (80E, 100 input neurons, 10% Poisson) fires at ~6% E rate with `exc_w=1.0`.
SORNMemory uses a larger network (300E, 512 input neurons, 11% structured input). With the
same `exc_w=1.0`, the 5x larger input fan-in gives ~5x more input current → 50-70% E rate.

**Fix:** Scale W_in (input weights) by 0.18 after creation. This normalizes the input drive
to match the original SORN's per-neuron current. In `create_episodic_memory`:
```julia
net.W_in.nzval .*= 0.18
```

## Adding a New Provider

1. Add struct `<: LLMProvider` with api_key, model, temperature, max_tokens
2. Add `create_*_provider()` constructor
3. Add `complete(provider::NewProvider, messages)` method
4. Add `test_connection(provider::NewProvider)` method
5. Add key parsing in `load_all_keys()` (match on key prefix in keys.txt)
6. Add to `select_provider()` in session.jl
7. Export from SORNMemory.jl
8. **Use abstract type signatures** in receiving functions (see "Debugging Gotchas" above)
