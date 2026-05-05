#include <vector>
#include <stdexcept>


void validateK(int K, int K1){
    if(K != K1){
        throw std::logic_error("K1 must be same value as K2 for matrix multiplication");
    }
}

std::vector<float> matmul(const std::vector<float>&A, int M, int K,
    const std::vector<float>&B,int K2, int N )
{
    validateK(K, K2);

    std::vector<float>C(M * N, 0.0f);
    // using row majority
    for(int i=0; i < M; ++i){
        for(int j=0; j < N; ++j){
            // for each row in i, multiply each col in j
            for(int k=0; k < K; ++k){
                C[i* N + j] += A[i * K + k] * B[k * N + j];
            };
        };
    };

    return C;
    
}
