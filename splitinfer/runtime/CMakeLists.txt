add_library(splitinfer_runtime
    src/manifest.cpp
    src/pipeline.cpp
    src/telemetry.cpp
    src/runtime.cpp
    src/gpu_executor.cpp
    src/fpga_executor.cpp
)
target_include_directories(splitinfer_runtime PUBLIC include)
target_link_libraries(splitinfer_runtime PRIVATE edgecoh)

# Optional TensorRT support — enable with -DSPLITINFER_HAS_TENSORRT=ON
option(SPLITINFER_HAS_TENSORRT "Build with TensorRT GPU executor support" OFF)
if(SPLITINFER_HAS_TENSORRT)
    find_package(CUDA REQUIRED)
    target_compile_definitions(splitinfer_runtime PRIVATE SPLITINFER_HAS_TENSORRT)
    target_include_directories(splitinfer_runtime PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(splitinfer_runtime PRIVATE nvinfer ${CUDA_LIBRARIES})
endif()

if(BUILD_TESTS)
    add_executable(test_manifest_rt tests/test_manifest.cpp)
    target_link_libraries(test_manifest_rt splitinfer_runtime)
    add_test(NAME test_manifest_rt COMMAND test_manifest_rt)

    add_executable(test_pipeline tests/test_pipeline_mock.cpp)
    target_link_libraries(test_pipeline splitinfer_runtime)
    add_test(NAME test_pipeline COMMAND test_pipeline)

    # Hardware integration test for TensorRT GPU executor.
    # Built unconditionally so the SKIP path is testable, but only does
    # actual GPU work when SPLITINFER_HAS_TENSORRT is defined.
    add_executable(test_gpu_executor tests/test_gpu_executor.cpp)
    target_link_libraries(test_gpu_executor splitinfer_runtime)
    if(SPLITINFER_HAS_TENSORRT)
        target_compile_definitions(test_gpu_executor PRIVATE SPLITINFER_HAS_TENSORRT)
    endif()
    # NOT registered with ctest — requires real GPU + pre-built engine file.
endif()

add_executable(splitinfer_run tools/splitinfer_run.cpp)
target_link_libraries(splitinfer_run splitinfer_runtime)
