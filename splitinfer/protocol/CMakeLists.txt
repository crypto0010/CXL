add_library(edgecoh
    src/messages.c
    src/edgecoh.c
    src/transport_usb.c
)
target_include_directories(edgecoh PUBLIC include)
find_library(LIBUSB usb-1.0)
if(LIBUSB)
    target_link_libraries(edgecoh PRIVATE ${LIBUSB})
endif()

if(BUILD_TESTS)
    enable_testing()
    add_executable(test_messages tests/test_messages.c)
    target_link_libraries(test_messages edgecoh)
    add_test(NAME test_messages COMMAND test_messages)

    add_executable(test_state_machine tests/test_state_machine.c)
    target_link_libraries(test_state_machine edgecoh)
    add_test(NAME test_state_machine COMMAND test_state_machine)

    # Integration test: requires Nexys 4 DDR FPGA connected via USB.
    # Built with BUILD_TESTS=ON but NOT registered with ctest (hardware required).
    add_executable(test_loopback tests/test_loopback.c)
    target_link_libraries(test_loopback edgecoh)
endif()
