add_library(splitinfer_runtime
    src/manifest.cpp
    src/pipeline.cpp
    src/telemetry.cpp
)
target_include_directories(splitinfer_runtime PUBLIC include)
target_link_libraries(splitinfer_runtime PRIVATE edgecoh)

if(BUILD_TESTS)
    add_executable(test_manifest_rt tests/test_manifest.cpp)
    target_link_libraries(test_manifest_rt splitinfer_runtime)
    add_test(NAME test_manifest_rt COMMAND test_manifest_rt)

    add_executable(test_pipeline tests/test_pipeline_mock.cpp)
    target_link_libraries(test_pipeline splitinfer_runtime)
    add_test(NAME test_pipeline COMMAND test_pipeline)
endif()

add_executable(splitinfer_run tools/splitinfer_run.cpp)
target_link_libraries(splitinfer_run splitinfer_runtime)
