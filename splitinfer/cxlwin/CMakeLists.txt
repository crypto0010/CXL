add_library(cxlwin
    src/cxlwin.c
    src/backend_emul.c
    src/backend_edgecoh.c
)
target_include_directories(cxlwin PUBLIC include)
target_link_libraries(cxlwin PUBLIC edgecoh PRIVATE pthread)

if(BUILD_TESTS)
    add_executable(test_cxlwin tests/test_cxlwin.c)
    target_link_libraries(test_cxlwin cxlwin)
    add_test(NAME test_cxlwin COMMAND test_cxlwin)
endif()
