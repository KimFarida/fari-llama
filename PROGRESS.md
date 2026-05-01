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
