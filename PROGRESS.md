# Progress Log

## [Date]
**Did:** 
**Learned:** 
**Stuck on:** 
**Tomorrow:**

## April 29 2026
**Did:** Scaffolded fari-llama repo, physio, SIWES, LC206 reversed linked list
**Learned:** Full project roadmap, in-place pointer reversal pattern
**Stuck on:** Memory allocation instinct — defaulting to new nodes instead of pointer manipulation
**Tomorrow:** Light day (physio), read Jay Alammar's Illustrated GPT-2 intro

## April 30 2026
**Did:** 
- Physio
- CMake build configured and running
- src/main.cpp compiles and executes (engine first boot yayyy!)
![Terminal Output](./images/Screenshot%202026-04-29%20at%2023.26.49.png)
- python/tokenizer.py skeleton created with encode/decode structure
- Read [HuggingFace BPE article](https://huggingface.co/learn/llm-course/chapter6/5) — understood merge rules and subword vocab
- LC21 Merge Two Sorted Lists (C++) — clean dummy head solution

**Learned:** 
- CMake project structure and build pipeline
- BPE tokenization — base vocab + merge rules, solves OOV (Out of Vocabulary) problem
- Dummy head trick for linked list problems
- Stack vs heap allocation for short-lived nodes

**Stuck on:** 
- Memory allocation instinct — defaulting to new nodes instead of pointer manipulation
- Byte Pair Encoding Tokenization in practice (will watch video to solidify concept)

**Tomorrow:** 
- Implement tokenizer encode() and decode() in Python
- Download TinyLlama vocabulary
- Validate against HuggingFace output