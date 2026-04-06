/* splitinfer/protocol/tests/test_messages.c */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "edgecoh/messages.h"

static void test_serialize_transfer_ownership(void) {
    edgecoh_transfer_msg_t msg = {
        .header = { .msg_type = EDGECOH_MSG_TRANSFER_OWNERSHIP, .flags = 0, .tensor_id = 42, .payload_len = 1 },
        .target_device = EDGECOH_DEV_FPGA,
    };
    uint8_t buf[64];
    int written = edgecoh_serialize(&msg, buf, sizeof(buf));
    assert(written == sizeof(edgecoh_transfer_msg_t));
    edgecoh_header_t hdr;
    int type = edgecoh_deserialize_header(buf, written, &hdr);
    assert(type == EDGECOH_MSG_TRANSFER_OWNERSHIP);
    assert(hdr.tensor_id == 42);
    assert(hdr.payload_len == 1);
    assert(buf[sizeof(edgecoh_header_t)] == EDGECOH_DEV_FPGA);
}

static void test_serialize_nmc_exec(void) {
    edgecoh_nmc_exec_msg_t msg = {
        .header = { .msg_type = EDGECOH_MSG_NMC_EXEC, .flags = 0, .tensor_id = 7,
                    .payload_len = sizeof(edgecoh_nmc_exec_msg_t) - sizeof(edgecoh_header_t) },
        .nmc_op = EDGECOH_NMC_EMBEDDING_LOOKUP, .table_base_addr = 0x00100000,
        .table_rows = 10000, .table_cols = 64, .input_addr = 0x00500000,
        .input_len = 128, .output_addr = 0x00600000,
    };
    uint8_t buf[128];
    int written = edgecoh_serialize(&msg, buf, sizeof(buf));
    assert(written == sizeof(edgecoh_nmc_exec_msg_t));
    edgecoh_header_t hdr;
    int type = edgecoh_deserialize_header(buf, written, &hdr);
    assert(type == EDGECOH_MSG_NMC_EXEC);
    assert(hdr.tensor_id == 7);
}

static void test_serialize_buffer_too_small(void) {
    edgecoh_transfer_msg_t msg = {
        .header = { .msg_type = EDGECOH_MSG_TRANSFER_OWNERSHIP, .flags = 0, .tensor_id = 1, .payload_len = 1 },
        .target_device = EDGECOH_DEV_HOST,
    };
    uint8_t buf[2];
    int written = edgecoh_serialize(&msg, buf, sizeof(buf));
    assert(written == -1);
}

static void test_deserialize_truncated(void) {
    uint8_t buf[4] = {EDGECOH_MSG_ACK, 0, 0, 0};
    edgecoh_header_t hdr;
    int type = edgecoh_deserialize_header(buf, 4, &hdr);
    assert(type == -1);
}

int main(void) {
    test_serialize_transfer_ownership();
    test_serialize_nmc_exec();
    test_serialize_buffer_too_small();
    test_deserialize_truncated();
    printf("All message tests passed.\n");
    return 0;
}
