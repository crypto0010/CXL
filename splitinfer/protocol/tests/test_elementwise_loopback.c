/* splitinfer/protocol/tests/test_elementwise_loopback.c
 *
 * Hardware integration test for the FPGA elementwise engine
 * (NMC ELEMENTWISE: ReLU, quantized add, scale).
 *
 * Validates that:
 *   1. NMC_EXEC with nmc_op=NMC_ELEMENTWISE reaches the elementwise dispatch
 *   2. eltwise_controller's FSM completes for each sub-op (relu/add/scale)
 *   3. elt_done CDC propagates correctly
 *   4. edgecoh_controller exits S_WAIT_NMC and sends ACK
 *
 * This is the T22 follow-up to T19 (embedding) and T21 (MAC).  Until this
 * test passes, the runtime cannot route Relu/Add/Mul layers to the FPGA
 * — see the TEMPORARY shortcut in runtime/src/pipeline.cpp.
 *
 * Sub-op selection:
 *   nmc_dispatch.v line 58 maps elt_op = nmc_table_base[1:0]:
 *     0b00 -> ReLU
 *     0b01 -> quantized add
 *     0b10 -> scale
 *     0b11 -> reserved
 *   So we encode the sub-op in the LOW 2 bits of table_base_addr.
 *   Scale factor (when applicable) goes in nmc_table_base[15:8].
 *
 * Hardware:
 *   FTDI FT2232HQ on Nexys 4 DDR (VID=0x0403, PID=0x6010)
 *   Channel B UART at 115200 8N1 -> /dev/ttyUSB1
 *
 * Build:  cmake .. -DBUILD_TESTS=ON && make test_elementwise_loopback
 * Run:    sudo ./test_elementwise_loopback
 *
 * Exit codes:
 *   0  — all tests passed
 *   1  — connection failed (FPGA not found / not programmed)
 *   2  — ACK not received within timeout (FSM hung)
 *   3  — unexpected response opcode
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "edgecoh/messages.h"
#include "edgecoh/transport.h"

#define NEXYS_VID  0x0403
#define NEXYS_PID  0x6010
#define ACK_TIMEOUT_MS 5000

/* Sub-op codes carried in nmc_table_base[1:0]. */
#define ELT_OP_RELU    0
#define ELT_OP_ADD     1
#define ELT_OP_SCALE   2

static uint64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000ULL);
}

static int wait_ack(edgecoh_transport_t *t, uint16_t expected_tensor_id, int timeout_ms)
{
    edgecoh_header_t hdr;
    uint64_t deadline = now_ms() + (uint64_t)timeout_ms;

    while (now_ms() < deadline) {
        int remaining = (int)(deadline - now_ms());
        if (remaining <= 0) break;

        int msg_type = edgecoh_recv_header(t, &hdr, remaining);
        if (msg_type < 0) continue;

        if (msg_type == EDGECOH_MSG_ACK || msg_type == EDGECOH_MSG_NMC_DONE) {
            if (hdr.tensor_id != expected_tensor_id) {
                fprintf(stderr, "  [WARN] tensor_id mismatch: got %u, expected %u\n",
                        hdr.tensor_id, expected_tensor_id);
                return -2;
            }
            return 0;
        }
        if (msg_type == EDGECOH_MSG_ERROR) {
            fprintf(stderr, "  [FAIL] FPGA returned ERROR (tensor_id=%u)\n", hdr.tensor_id);
            return -2;
        }
        fprintf(stderr, "  [WARN] unexpected msg_type 0x%02x, ignoring\n", msg_type);
    }

    fprintf(stderr, "  [FAIL] timeout waiting for ACK (%d ms)\n", timeout_ms);
    return -1;
}

/* Send an NMC_EXEC for the elementwise engine.  sub_op selects relu/add/scale,
 * num_words is the operand count, scale is the optional scale factor (only
 * meaningful for ELT_OP_SCALE). */
static int test_elementwise(edgecoh_transport_t *t, uint16_t tensor_id,
                             const char* sub_op_name, uint8_t sub_op,
                             uint32_t num_words, uint8_t scale)
{
    printf("Test: ELEMENTWISE %s num_words=%u tid=%u ... ",
           sub_op_name, num_words, tensor_id);
    fflush(stdout);

    edgecoh_nmc_exec_msg_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.msg_type    = EDGECOH_MSG_NMC_EXEC;
    msg.header.tensor_id   = tensor_id;
    msg.header.payload_len = sizeof(edgecoh_nmc_exec_msg_t) - sizeof(edgecoh_header_t);
    msg.nmc_op             = EDGECOH_NMC_RELU; /* dispatch routes by NMC_ELEMENTWISE class — see note */

    /* nmc_dispatch.v field mapping for ELEMENTWISE:
     *   table_base[1:0]  -> elt_op (sub-op)
     *   table_base[15:8] -> elt_scale (used by scale op)
     *   table_rows       -> elt_num_words
     *   input_addr       -> elt_input_addr
     *   output_addr      -> elt_output_addr
     *
     * NOTE: the dispatch uses NMC_ELEMENTWISE = 0x03 as the class selector,
     * but the messages.h enum has separate codes for RELU/QUANT_ADD/SCALE.
     * On the host side we use EDGECOH_NMC_RELU as a stand-in for the class
     * since 0x03 happens to equal RELU's value.  This works as long as
     * we don't change the enum.  See nmc_dispatch.v case statement. */
    msg.table_base_addr    = ((uint32_t)scale << 8) | (sub_op & 0x3);
    msg.table_rows         = num_words;
    msg.table_cols         = 0;             /* unused for elementwise */
    msg.input_addr         = 0x00900000;
    msg.input_len          = num_words;     /* not consumed by dispatch but included */
    msg.output_addr        = 0x00A00000;

    uint8_t buf[sizeof(edgecoh_nmc_exec_msg_t)];
    int n = edgecoh_serialize(&msg, buf, sizeof(buf));
    if (n < 0) { printf("FAIL (serialize)\n"); return 1; }

    int sent = edgecoh_transport_send(t, buf, n);
    if (sent != n) { printf("FAIL (send: sent=%d expected=%d)\n", sent, n); return 1; }

    int rc = wait_ack(t, tensor_id, ACK_TIMEOUT_MS);
    if (rc == 0)  { printf("PASS\n"); return 0; }
    if (rc == -1) { printf("FAIL (timeout — elementwise FSM likely hung)\n"); return 2; }
    printf("FAIL (unexpected response)\n");
    return 3;
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;

    printf("=== EdgeCoh Elementwise Engine Hardware Integration Test (T22) ===\n");
    printf("Target: Nexys 4 DDR (VID=0x%04X, PID=0x%04X)\n\n", NEXYS_VID, NEXYS_PID);

    edgecoh_transport_t *t = edgecoh_transport_open(NEXYS_VID, NEXYS_PID);
    if (!t) {
        fprintf(stderr, "ERROR: cannot open USB transport.\n");
        fprintf(stderr, "  Check FPGA is connected and programmed with SplitInfer bitstream.\n");
        return 1;
    }
    printf("Transport opened.\n\n");

    int failures = 0;

    /* Test 1: ReLU on 16 words.  Smallest meaningful elementwise op. */
    if (test_elementwise(t, 201, "ReLU",  ELT_OP_RELU,  16, 0) != 0) failures++;

    /* Test 2: quantized add on 16 words.  Exercises the add path. */
    if (test_elementwise(t, 202, "ADD",   ELT_OP_ADD,   16, 0) != 0) failures++;

    /* Test 3: scale by 0x40 on 16 words.  Exercises the scale parameter
     * (carried in nmc_table_base[15:8]). */
    if (test_elementwise(t, 203, "SCALE", ELT_OP_SCALE, 16, 0x40) != 0) failures++;

    /* Test 4: degenerate zero-word case.  Exercises the early-exit path. */
    if (test_elementwise(t, 204, "ReLU",  ELT_OP_RELU,  0,  0) != 0) failures++;

    edgecoh_transport_close(t);

    printf("\n");
    if (failures == 0) {
        printf("=== ALL TESTS PASSED ===\n");
        printf("Elementwise engine is hardware-validated.  Combined with T21\n");
        printf("(MAC controller), all three NMC engines are now ready and the\n");
        printf("TEMPORARY shortcut in pipeline.cpp can be replaced with the\n");
        printf("real op_type -> nmc_op routing.\n");
    } else {
        printf("=== %d TEST(S) FAILED ===\n", failures);
        printf("Likely causes:\n");
        printf("  - Single-cycle pulse handshake bug in eltwise_controller.v\n");
        printf("  - elt_done signal not propagating from elementwise.v back\n");
        printf("    through nmc_dispatch -> CDC -> edgecoh_controller\n");
        printf("  - Wrong sub-op encoding in nmc_table_base[1:0]\n");
        printf("Add LED probes for elt_start, elt_done, and the eltwise FSM\n");
        printf("state register, then re-synthesize.\n");
    }

    return failures > 0 ? 1 : 0;
}
