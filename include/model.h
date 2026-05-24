#pragma once
#include <vector>
#include <string>


struct TinyLlamaWeights{
    std::vector<float> embed_tokens; // [32000 * 32048]

    int vocab_size = 32000;
    int hidden_size = 2048;
    int num_layers = 22;
};


std::vector<float> embedding_lookup(const TinyLlamaWeights& weights, int token_id);

void load_weights(TinyLlamaWeights& weights, const std::string& model_dir);

std:: vector<float> rms_norm(
    const std::vector<float>& x,
    const std::vector<float>& weight,
    float epsilon
);