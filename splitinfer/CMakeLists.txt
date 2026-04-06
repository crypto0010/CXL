cmake_minimum_required(VERSION 3.18)
project(splitinfer C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS "Build tests" ON)

add_subdirectory(protocol)
# runtime subdirectory added in Phase 4
add_subdirectory(runtime)
