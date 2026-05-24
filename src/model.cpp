#include "../include/model.h"
#include <fstream>
#include <stdexcept>
#include <iostream>


// Lookup table function, gets the row based on token id from embedded table
std::vector<float> embedding_lookup(
    const TinyLlamaWeights& weights,
    int token_id

){
    int n = weights.hidden_size;
    int start = token_id * n;
    std::vector<float> res(n);

    for(int i=0; i < n; i++){
        res[i] = weights.embed_tokens[start + i];
    }

    return res;
};

void load_weights(TinyLlamaWeights& weights, const std::string& model_dir){
    std::string path = model_dir + "/embed_tokens.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    int total = weights.vocab_size * weights.hidden_size;
    weights.embed_tokens.resize(total);
    f.read(reinterpret_cast<char*>(weights.embed_tokens.data()), total * sizeof(float));

    if (!f) throw std::runtime_error("Failed to read embed_tokens.bin");
    std::cout << "Loaded embed_tokens: " << total << " floats\n";

}