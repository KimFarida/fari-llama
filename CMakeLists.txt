cmake_minimum_required(VERSION 3.16)
project(fari-llama VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main executable
add_executable(fari-llama
    src/main.cpp
)

# Headers
target_include_directories(fari-llama PRIVATE include)