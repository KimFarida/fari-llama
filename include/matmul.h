#pragma once
#include <vector>

std::vector<float> matmul(const std::vector<float>& A, int M, int K,
                           const std::vector<float>& B, int K2, int N);