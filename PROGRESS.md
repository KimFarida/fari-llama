# Progress Log

## [Date]
**Did:** 
**Learned:** 
**Stuck on:** 
**Tomorrow:**

## April 28 2026
**Did:** 
- Scaffolded fari-llama repo, physio, SIWES, LC206 reversed linked list

**Learned:**
- Full project roadmap, in-place pointer reversal pattern

**Stuck on:**
- Memory allocation instinct — defaulting to new nodes instead of pointer manipulation

**Tomorrow:** Light day (physio), read [Jay Alammar's Illustrated GPT-2 intro](https://jalammar.github.io/illustrated-gpt2/)

## April 29 2026
**Did:** 
- Physio
- CMake build configured and running
- src/main.cpp compiles and executes (engine first boot yayyy!)
![Terminal Output](./images/Screenshot%202026-04-29%20at%2023.26.49.png)
- python/tokenizer.py skeleton created with encode/decode structure
- Read [HuggingFace BPE article](https://huggingface.co/learn/llm-course/chapter6/5) — understood merge rules and subword vocab
- LC21 Merge Two Sorted Lists (C++) — clean dummy head solution

**Learned:** 
- GPT-2 architecture (Jay Alammar): `tokens` → `embeddings` → 
  `self-attention (Q/K/V)` → `masked decoding` → `softmax` → 
  `top-k sampling`.
  *Model never sees words, only token IDs.*
  `Context window` = max tokens visible at once (2048 for TinyLlama)
- CMake project structure and build pipeline
- BPE tokenization — base vocab + merge rules, solves OOV (Out of Vocabulary) problem
- Dummy head trick for linked list problems
- Stack vs heap allocation for short-lived nodes

**Stuck on:** 
- Byte Pair Encoding Tokenization in practice (will watch video to solidify concept)

**Tomorrow:** 
- Implement tokenizer encode() and decode() in Python
- Download TinyLlama vocabulary
- Validate against HuggingFace output


## April 30 2026
**Did:** Python tokenizer implemented and validated, CMake 
build running, LC143 Reorder List solved in C++

**Learned:** 
- HTTP request lifecycle (DNS→TCP→TLS→request→response)
HTTP Request how it works
DNS Resolution -> 
TCP Handshake ->
TLS Hnadshake ->
HTTP Request(Method, Path, Headers(metadata), Body)->
Server Processing ->
HTTP Response (Status, Headers, Body)-> 
Connection Handling 1.0/1.1(keep-alive)/1.2(Parrallel)/1.3(UDP)
```
Client          DNS          Server
  |              |              |
  |--DNS query-->|              |
  |<--IP addr----|              |
  |                             |
  |--------TCP SYN------------->|
  |<-------TCP SYN-ACK----------|
  |--------TCP ACK------------->|
  |                             |
  |--------TLS handshake------->|
  |<-------TLS handshake--------|
  |                             |
  |--------HTTP Request-------->|
  |<-------HTTP Response--------|
```
- BPE subword tokenization
![Terminal Output](./images/Screenshot%202026-04-30%20at%2013.48.14.png)
- fast/slow pointer for finding list middle, three-step reorder pattern (split + reverse + merge)

**Stuck on:** merge() design — return value vs void, 
cleaner pointer weaving without counter

**Tomorrow:** Begin C++ tokenizer header, intro to 
transformer weight structure

## May 1 2026
**Did:**
- Created C++ tokenizer header (include/tokenizer.h)
- LC238 Product of Array Except Self — Python, both O(n) 
  space and O(1) space solutions

**Learned:**
- `explicit` keyword — prevents implicit constructor conversion
- unordered_map vs map — O(1) vs O(log n) lookup, speed matters 
  at inference scale
- Bidirectional maps — encode and decode need separate 
  O(1) lookups in opposite directions
- KV Cache — saves K,V pairs from previous tokens to avoid 
  recomputation, append-only within a single sequence
- LRU/LFU/FIFO eviction — relevant for multi-sequence cache 
  management in Phase 6
- Prefix/suffix product pattern — O(1) space by using output 
  array as running product store

**Stuck on:**
- KV Cache eviction context — confused on single-sequence 
  (append-only) with multi-sequence (eviction needed)

**Tomorrow:**
- Rest day or light reading

## May 3 2026
**Did:**
- LC226 Invert Binary Tree — recursive DFS, Python
- LC104 added to trees folder
- Transformer weight structure — full mental model

**Learned:**
- TinyLlama has 22 layers, each with attention + MLP + norm
- Embedding table [32000 x 2048] — one vector per token
- Q/K/V separation — different roles in attention mechanism
- Layer norm — prevents numerical explosion between layers
- Embedding dimension (2048) flows through entire model

**Stuck on:**
- Q/K/V intuition — know they're separate, still building 
  feel for what each one does geometrically

**Next week:**
- Load TinyLlama weights in Python
- Implement matrix multiply in C++
- Start attention forward pass

## May 20 2026
**Did:**
- Downloaded TinyLlama weights (model.safetensors, 2.2GB)
- Wrote python/inspect_weights.py — confirmed all weight shapes
- Exported embed_tokens to binary (python/export_weights.py)
- Implemented matmul in C++ with row-major indexing
- Created include/model.h and src/model.cpp
- Implemented embedding_lookup — loads and indexes flat weight vector
- Implemented rms_norm — RMSNorm formula from scratch
- Loaded real weights into C++ and verified embedding lookup output
- LC572 Subtree of Another Tree — fixed dfs + isSameTree separation
- LC15 3Sum — two pointer with duplicate skipping

**Learned:**
- GQA (Grouped Query Attention) — K/V are [256x2048] not [2048x2048]
  because only 4 K/V heads shared across 32 Q heads. K/V cached,
  Q is not — so shrinking K/V saves memory at inference
- Row-major indexing — element (i,j) in matrix of width K is at [i*K + j]
- reinterpret_cast — reading raw binary floats from file into vector
- Embedding table is fixed after training, BPE guarantees no OOV
- RMSNorm formula: y = (x / sqrt(mean(x²) + ε)) * weight

**Stuck on:**
- Full forward pass linearity — how all pieces connect end to end

**Tomorrow:**
- Attention mechanism — Q, K, V projections (three matmuls)
- Export remaining weights from Python

## May 21 - May 26 2026
**Did:**
- Exported attention weights to binary (q_proj, k_proj, v_proj, o_proj for layer 0)
- Loaded all attention weights into C++ TinyLlamaWeights struct
- Implemented attention_forward — Q/K/V projections, GQA head loop, output projection
- Understood single head attention first, then generalized to all 32 heads

**Learned:**
- GQA group size = num_heads / num_kv_heads = 8, every 8 Q heads share 1 K/V head
- Attention scores scaled by 1/sqrt(head_dim) to prevent softmax saturation
- o_proj mixes information across all heads into single coherent representation
- Single token attention simplifies — no softmax needed, just scale V by score

**Stuck on:**
- Nothing blocking, forward pass linearity is clearer now

**Tomorrow:**
- MLP forward pass (gate_proj, up_proj, down_proj)
- Export MLP weights from Python
- Wire attention + MLP into single layer forward pass