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

    # T21: MAC controller hardware validation (NMC INT8 FC engine).
    # Sends NMC_EXEC with nmc_op=NMC_INT8_FC and verifies the FPGA's
    # mac_controller FSM completes and returns an ACK.
    add_executable(test_mac_loopback tests/test_mac_loopback.c)
    target_link_libraries(test_mac_loopback edgecoh)

    # T22: elementwise engine hardware validation (NMC ELEMENTWISE).
    # Sends NMC_EXEC with nmc_op=NMC_ELEMENTWISE and verifies the FPGA's
    # eltwise_controller FSM completes and returns an ACK.
    add_executable(test_elementwise_loopback tests/test_elementwise_loopback.c)
    target_link_libraries(test_elementwise_loopback edgecoh)
endif()
