#include <iostream>
#include "matmul.h"
#include "model.h"


int main(){
    std::cout << "fari-llama: inference engine initializing..." << std::endl;
    std::cout << "Version 0.1.0"<<std::endl;

    TinyLlamaWeights weights;
    load_weights(weights, "models/tinyllama");

    auto embedding = embedding_lookup(weights, 1);
    std::cout << "Embedding[0:5]: ";
    for(int i = 0; i < 5; i++) std::cout << embedding[i] << " ";
    std::cout <<std::endl;

    return 0; 
}
