#include "model.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cmath>


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

// Load weights in binary form to c++
void load_weights(TinyLlamaWeights& weights, const std::string& model_dir){
    // embed_tokens: 32000 x 2048
    {
        std::string path = model_dir + "/embed_tokens.bin";
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open " + path);
        int total = weights.vocab_size * weights.hidden_size;
        weights.embed_tokens.resize(total);
        f.read(reinterpret_cast<char*>(weights.embed_tokens.data()), total * sizeof(float));
        std::cout << "Loaded embed_tokens: " << total << " floats\n";
    }
    
    // q_proj_0: 2048 x 2048
    {
        std::string path = model_dir + "/q_proj_0.bin";
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open " + path);
        int total = weights.hidden_size * weights.hidden_size;
        weights.q_proj_0.resize(total);
        f.read(reinterpret_cast<char*>(weights.q_proj_0.data()), total * sizeof(float));
        std::cout << "Loaded q_proj_0: " << total << " floats\n";
    }
    
    // k_proj_0: 256 x 2048
    {
         std::string path = model_dir + "/k_proj_0.bin";
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open " + path);
        int total = (weights.num_kv_heads * weights.head_dim) * weights.hidden_size;
        weights.k_proj_0.resize(total);
        f.read(reinterpret_cast<char*>(weights.k_proj_0.data()), total * sizeof(float));
        std::cout << "Loaded k_proj_0: " << total << " floats\n";
    }
    
    
    // v_proj_0: 256 x 2048 
    {
        std::string path = model_dir + "/v_proj_0.bin";
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open " + path);
        int total = (weights.num_kv_heads * weights.head_dim) * weights.hidden_size;
        weights.v_proj_0.resize(total);
        f.read(reinterpret_cast<char*>(weights.v_proj_0.data()), total * sizeof(float));
        std::cout << "Loaded v_proj_0: " << total << " floats\n";
    }

    // o_proj_0: 2048 x 2048
    {
        std::string path = model_dir + "/o_proj_0.bin";
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open " + path);
        int total =  weights.hidden_size * weights.hidden_size;
        weights.o_proj_0.resize(total);
        f.read(reinterpret_cast<char*>(weights.o_proj_0.data()), total * sizeof(float));
        std::cout << "Loaded o_proj_0: " << total << " floats\n";
    }
}


std:: vector<float> rms_norm(
    const std::vector<float>& x,
    const std::vector<float>& weight,
    float epsilon = 1e-5f
){

    // square every element in x
    float sum_of_squares = 0;
    int size = x.size();

    for(int i = 0; i < size; i++) sum_of_squares += x[i] * x[i];

    float mean_squares = sum_of_squares/size;
    float z = sqrt(mean_squares + epsilon);

    std::vector<float> y(size);

    for(int i = 0; i < size; i++) y[i] =  (x[i]/z) *weight[i];
    return y;

};