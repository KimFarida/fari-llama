PHASE 1: Tokenizer (Week 1-2)
├── Load a pretrained BPE vocabulary (from TinyLlama)
├── Implement encode() — string → list of token IDs
├── Implement decode() — token IDs → string
└── Validate against HuggingFace tokenizer output

PHASE 2: Model Loading (Week 2)
├── Download TinyLlama weights (.bin or .gguf format)
├── Parse and load weights into C++ data structures
└── Understand weight shapes — embedding, attention, MLP layers

PHASE 3: Forward Pass in C++ (Week 3-4)
├── Matrix multiply (the core operation)
├── Embedding lookup
├── Layer norm
├── Attention (Q, K, V — the interesting part)
├── MLP block
└── Softmax + sampling (greedy first, then temperature)

PHASE 4: KV Cache (Week 4-5)
├── Understand WHY it exists (speed — avoids recomputing)
├── Implement the cache structure
└── Integrate into attention forward pass

PHASE 5: HTTP Server in Go (Week 5-6)
├── Simple REST endpoint — POST /generate
├── Accepts prompt, returns completion
└── Connects Go server to C++ inference core via subprocess or CGO

PHASE 6: Continuous Batching (Week 6-7)
├── Handle multiple requests simultaneously
├── Implement basic scheduler
└── Benchmark throughput vs naive approach

PHASE 7: Benchmarks + Polish (Week 8)
├── Compare token/sec vs llama.cpp on same hardware
├── Clean README with architecture diagram
├── One blog post explaining what I built
└── Demo video or GIF for GitHub