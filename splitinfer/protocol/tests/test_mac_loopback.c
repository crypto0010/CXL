/* splitinfer/protocol/tests/test_mac_loopback.c
 *
 * Hardware integration test for the FPGA MAC controller (NMC INT8 FC engine).
 *
 * Validates that:
 *   1. NMC_EXEC with nmc_op=NMC_INT8_FC reaches the MAC controller dispatch
 *   2. mac_controller's S_IDLE -> S_CLEAR -> ... -> S_DONE FSM completes
 *   3. emb_done CDC to nmc_done_sys propagates correctly
 *   4. edgecoh_controller exits S_WAIT_NMC and sends ACK
 *
 * This is the T21 follow-up to T19 (which only validated the embedding
 * lookup engine).  Until this test passes, the runtime cannot route
 * MatMul/Gemm layers to the FPGA — see the TEMPORARY shortcut in
 * runtime/src/pipeline.cpp's op_type_to_nmc_op().
 *
 * Hardware:
 *   FTDI FT2232HQ on Nexys 4 DDR (VID=0x0403, PID=0x6010)
 *   Channel B UART at 115200 8N1 -> /dev/ttyUSB1
 *
 * Build (only when FPGA is connected):
 *   cmake .. -DBUILD_TESTS=ON && make test_mac_loopback
 *
 * Run (requires sudo for tty access):
 *   sudo ./test_mac_loopback
 *
 * Exit codes:
 *   0  — all tests passed
 *   1  — connection failed (FPGA not found / not programmed)
 *   2  — ACK not received within timeout (likely hang in MAC FSM)
 *   3  — unexpected response opcode
 *
 * Test parameters:
 *   We send minimal MAC parameters (M=K=8 = one tile of the 8x8 array)
 *   pointing at uninitialized DDR2 memory.  Like T19's embedding test,
 *   we don't care about numerical correctness — only that the FSM
 *   terminates and acknowledges.  If the MAC controller hangs, we know
 *   there's a handshake bug analogous to the ones we already fixed in
 *   edgecoh_controller's S_SEND_ACK and ddr2_arbiter's pulse-drop path.
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

/* Generous timeout — MAC operations on uninitialized DDR2 should still
 * complete in microseconds, but we allow plenty of slack for first run. */
#define ACK_TIMEOUT_MS 5000

static uint64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000ULL);
}

/* Wait for an ACK with the expected tensor_id, or timeout. */
static int wait_ack(edgecoh_transport_t *t, uint16_t expected_tensor_id, int timeout_ms)
{
    edgecoh_header_t hdr;
    uint64_t deadline = now_ms() + (uint64_t)timeout_ms;

    while (now_ms() < deadline) {
        int remaining = (int)(deadline - now_ms());
        if (remaining <= 0) break;

        int msg_type = edgecoh_recv_header(t, &hdr, remaining);
        if (msg_type < 0) continue;

        /* Accept either the universal MSG_ACK (current firmware) or
         * the reserved MSG_NMC_DONE (future firmware) — see messages.h. */
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

/* Send an NMC_EXEC for the MAC engine (NMC_INT8_FC) with given M and K dims.
 * Address fields point at safe (but uninitialized) DDR2 regions; we don't
 * verify output correctness, only that the FSM terminates with an ACK. */
static int test_mac_int8_fc(edgecoh_transport_t *t, uint16_t tensor_id,
                             uint32_t M, uint32_t K)
{
    printf("Test: NMC_INT8_FC M=%u K=%u tid=%u ... ", M, K, tensor_id);
    fflush(stdout);

    edgecoh_nmc_exec_msg_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.msg_type    = EDGECOH_MSG_NMC_EXEC;
    msg.header.tensor_id   = tensor_id;
    msg.header.payload_len = sizeof(edgecoh_nmc_exec_msg_t) - sizeof(edgecoh_header_t);
    msg.nmc_op             = EDGECOH_NMC_INT8_FC;
    /* nmc_dispatch.v field mapping for INT8_FC:
     *   table_base_addr -> mac_weight_addr (DDR2 base of [M x K] weights)
     *   table_rows      -> mac_M (output rows)
     *   table_cols      -> mac_K (inner dimension)
     *   input_addr      -> mac_input_addr (activation vector base)
     *   output_addr     -> mac_output_addr (result vector base) */
    msg.table_base_addr    = 0x00200000;  /* well away from embedding test's 0x00100000 */
    msg.table_rows         = M;
    msg.table_cols         = K;
    msg.input_addr         = 0x00700000;
    msg.input_len          = K;            /* not consumed by MAC; included for completeness */
    msg.output_addr        = 0x00800000;

    uint8_t buf[sizeof(edgecoh_nmc_exec_msg_t)];
    int n = edgecoh_serialize(&msg, buf, sizeof(buf));
    if (n < 0) { printf("FAIL (serialize)\n"); return 1; }

    int sent = edgecoh_transport_send(t, buf, n);
    if (sent != n) { printf("FAIL (send: sent=%d expected=%d)\n", sent, n); return 1; }

    int rc = wait_ack(t, tensor_id, ACK_TIMEOUT_MS);
    if (rc == 0)  { printf("PASS\n"); return 0; }
    if (rc == -1) { printf("FAIL (timeout — MAC FSM likely hung)\n"); return 2; }
    printf("FAIL (unexpected response)\n");
    return 3;
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;

    printf("=== EdgeCoh MAC Controller Hardware Integration Test (T21) ===\n");
    printf("Target: Nexys 4 DDR (VID=0x%04X, PID=0x%04X)\n\n", NEXYS_VID, NEXYS_PID);

    edgecoh_transport_t *t = edgecoh_transport_open(NEXYS_VID, NEXYS_PID);
    if (!t) {
        fprintf(stderr, "ERROR: cannot open USB transport.  Check that:\n");
        fprintf(stderr, "  1. The Nexys 4 DDR is connected via USB\n");
        fprintf(stderr, "  2. /dev/ttyUSB1 exists and is readable\n");
        fprintf(stderr, "  3. The FPGA is programmed with the SplitInfer bitstream\n");
        return 1;
    }
    printf("Transport opened.\n\n");

    int failures = 0;

    /* Test 1: smallest valid case — single 8x8 tile (one M iteration,
     * one K iteration).  Exercises the basic FSM path. */
    if (test_mac_int8_fc(t, /*tid=*/101, /*M=*/8, /*K=*/8) != 0) failures++;

    /* Test 2: zero dimensions — exercises the early-exit path
     * (S_NEXT_M with M=0 should jump to S_DONE).  This is a degenerate
     * case but tests that the FSM doesn't deadlock on edge inputs. */
    if (test_mac_int8_fc(t, /*tid=*/102, /*M=*/0, /*K=*/0) != 0) failures++;

    /* Test 3: medium case — 16x16 to exercise multiple m_idx iterations
     * (16 = 2 outer rows x 8 inner steps per row).  Catches bugs in
     * the m_idx and k_idx counter logic. */
    if (test_mac_int8_fc(t, /*tid=*/103, /*M=*/16, /*K=*/16) != 0) failures++;

    edgecoh_transport_close(t);

    printf("\n");
    if (failures == 0) {
        printf("=== ALL TESTS PASSED ===\n");
        printf("MAC controller is hardware-validated.  You can now restore\n");
        printf("per-op routing in runtime/src/pipeline.cpp by replacing the\n");
        printf("TEMPORARY return 0x01 with the real op_type mapping.\n");
    } else {
        printf("=== %d TEST(S) FAILED ===\n", failures);
        printf("Likely causes:\n");
        printf("  - Single-cycle pulse handshake bug in mac_controller.v\n");
        printf("    (similar to the S_SEND_ACK / arbiter bugs we fixed earlier)\n");
        printf("  - mac_array_8x8 mac_done signal not propagating correctly\n");
        printf("  - DDR2 read timing issue exposed by uninitialized memory access\n");
        printf("Investigate by adding LED probes for mac_start, mac_done, and\n");
        printf("the mac_controller's state register, then re-synthesize.\n");
    }

    return failures > 0 ? 1 : 0;
}
