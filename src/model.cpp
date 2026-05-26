#include "model.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include "matmul.h"


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


// Rms = (x / sqrt(mean(x²) + ε)) * weight
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


std::vector<float> attention_forward(
        const TinyLlamaWeights& weights,
        const std::vector<float>& x  // input token embedding x [1*2048]
){
    // Project input into QKV using learned weights
    // token x * QKVO

    // [1*2048] x [2048*2048] (32 heads x 64 head_dim)
    auto Q = matmul(x, 1, weights.hidden_size, weights.q_proj_0,  weights.hidden_size,  weights.hidden_size);

    // [1*2048] x [256x2048] (4 kv_heads x 64 head_dim)
    auto K = matmul(x, 1, weights.hidden_size, weights.k_proj_0,  weights.hidden_size,  weights.num_kv_heads * weights.head_dim);
    auto V = matmul(x, 1, weights.hidden_size, weights.v_proj_0, weights.hidden_size, weights.num_kv_heads * weights.head_dim);

    // attention score - Q * K ^ T/ sqrt(head_dim)
    // scaling to so values not too large
    float scale = 1.0f / std::sqrt((float)weights.head_dim);

    // Recall 32 heads at Q, 4 at K so 8Q:1K
    int group_size = weights.num_heads / weights.num_kv_heads;

    std::vector<float> all_heads_out(weights.hidden_size, 0.0f);

    for (int h =0; h < weights.num_heads; h++){
        // head_dim -> 64 i.e head0 : 0:64, 64:128...
        int q_offset = h * weights.head_dim;

        // 8 Qheads to 1 KV
        // kv_head0 ->heads 0-7, kv_head1 ->heads 8-15..
        int kv_head = h / group_size;
        int kv_offset = kv_head * weights.head_dim;

        //  Dot product of Q head and K head -> attention score current head
        // how relevant this token's query is to its own key
        float score = 0.0f;
        int j = kv_offset;
        // for every i+64 q, used a k head to calculate 
        for(int i = q_offset; i < q_offset + weights.head_dim; i++){
            score += Q[i] * K[j];
            j++;
            
        }
        score*=scale;

        // Weight V by attention score -> current head contribution to output
        // High score means V passes through strongly
        int k = kv_offset;
        for(int i = q_offset; i < q_offset + weights.head_dim; i++){
            all_heads_out[i] = score * V[k];
            k++;
        }

        
    
    }

    // all head outputs [2048] projected back to hidden space [2048]
    // o_proj mixes information across heads into a single representation
    auto out = matmul(all_heads_out, 1, weights.hidden_size, weights.o_proj_0, weights.hidden_size, weights.hidden_size);
    return out;
}