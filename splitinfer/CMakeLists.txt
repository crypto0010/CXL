cmake_minimum_required(VERSION 3.18)
project(splitinfer C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS "Build tests" ON)

# Enable CTest at the top level so add_test() calls in subdirectories are
# aggregated into build/CTestTestfile.cmake.  Without this, `ctest` from the
# top-level build dir reports "No tests were found" even though subdirs
# define tests, because the per-subdir CTestTestfile.cmake files are never
# included by a top-level one.
if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(protocol)
add_subdirectory(runtime)
