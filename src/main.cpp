#include <iostream>
#include <string>

std::vector<float> matmul(const std::vector<float>& A, int M, int K,
                           const std::vector<float>& B, int K2, int N);

int main(){
    std::cout << "fari-llama: inference engine initializing..." << std::endl;
    std::cout << "Version 0.1.0"<<std::endl;


    std::vector<float> A = {1,2,3,4};
    std::vector<float> B = {5,6,7,8};
    
    auto C = matmul(A, 2, 2, B, 2, 2);
    for (float v : C) std::cout << v << " ";
    std::cout << std::endl;
    return 0;

}
