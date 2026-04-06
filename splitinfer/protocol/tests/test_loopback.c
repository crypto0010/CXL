/* splitinfer/protocol/tests/test_loopback.c
 *
 * Integration Test: USB Loopback with Real FPGA (Nexys 4 DDR)
 *
 * Sends SYNC_BARRIER and NMC_EXEC messages to the FPGA over USB and expects
 * an ACK response for each.  Requires the Nexys 4 DDR board to be connected
 * via USB and programmed with the SplitInfer bitstream.
 *
 * Hardware:
 *   FTDI FT2232HQ: VID=0x0403, PID=0x6010
 *   Channel B (UART) — EdgeCoh byte-stream protocol
 *
 * Build (NOT added to ctest — requires hardware):
 *   cmake .. -DBUILD_TESTS=ON && make test_loopback
 *
 * Run:
 *   ./test_loopback
 *
 * Exit codes:
 *   0 — all tests passed
 *   1 — connection failed (FPGA not found or not programmed)
 *   2 — ACK not received within timeout
 *   3 — unexpected response
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "edgecoh/messages.h"
#include "edgecoh/transport.h"

/* FTDI identifiers for Nexys 4 DDR (FT2232HQ) */
#define NEXYS_VID  0x0403
#define NEXYS_PID  0x6010

/* Timeout for waiting for ACK from FPGA (milliseconds) */
#define ACK_TIMEOUT_MS 2000

/* -------------------------------------------------------------------------
 * Utility: wall-clock milliseconds
 * ---------------------------------------------------------------------- */
static uint64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000ULL);
}

/* -------------------------------------------------------------------------
 * wait_ack: poll transport until ACK arrives or timeout.
 * Returns 0 on success, -1 on timeout, -2 on error/unexpected response.
 * ---------------------------------------------------------------------- */
static int wait_ack(edgecoh_transport_t *transport,
                    uint16_t expected_tensor_id,
                    int timeout_ms)
{
    edgecoh_header_t hdr;
    uint64_t deadline = now_ms() + (uint64_t)timeout_ms;

    while (now_ms() < deadline) {
        int remaining_ms = (int)(deadline - now_ms());
        if (remaining_ms <= 0) break;

        int msg_type = edgecoh_recv_header(transport, &hdr, remaining_ms);
        if (msg_type < 0) {
            /* Timeout or temporary read error — keep trying */
            continue;
        }

        if (msg_type == EDGECOH_MSG_ACK) {
            if (hdr.tensor_id != expected_tensor_id) {
                fprintf(stderr, "  [WARN] ACK tensor_id mismatch: got %u, expected %u\n",
                        hdr.tensor_id, expected_tensor_id);
                return -2;
            }
            return 0;  /* success */
        }

        if (msg_type == EDGECOH_MSG_ERROR) {
            fprintf(stderr, "  [FAIL] FPGA returned ERROR message (tensor_id=%u)\n",
                    hdr.tensor_id);
            return -2;
        }

        fprintf(stderr, "  [WARN] Unexpected message type 0x%02x, ignoring\n", msg_type);
    }

    fprintf(stderr, "  [FAIL] Timeout waiting for ACK (%d ms)\n", timeout_ms);
    return -1;
}

/* -------------------------------------------------------------------------
 * Test 1: SYNC_BARRIER -> ACK
 * ---------------------------------------------------------------------- */
static int test_sync_barrier(edgecoh_transport_t *transport) {
    printf("Test 1: SYNC_BARRIER -> ACK ... ");
    fflush(stdout);

    edgecoh_barrier_msg_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_type    = EDGECOH_MSG_SYNC_BARRIER;
    msg.flags       = 0;
    msg.tensor_id   = 1;
    msg.payload_len = 0;

    uint8_t tx_buf[sizeof(edgecoh_barrier_msg_t)];
    int n = edgecoh_serialize(&msg, tx_buf, sizeof(tx_buf));
    if (n < 0) {
        printf("FAIL (serialize error)\n");
        return 1;
    }

    int sent = edgecoh_transport_send(transport, tx_buf, n);
    if (sent != n) {
        printf("FAIL (send error: sent=%d expected=%d)\n", sent, n);
        return 1;
    }

    int rc = wait_ack(transport, /*expected_tensor_id=*/1, ACK_TIMEOUT_MS);
    if (rc == 0)  { printf("PASS\n"); return 0; }
    if (rc == -1) { printf("FAIL (timeout)\n"); return 2; }
    printf("FAIL (unexpected response)\n");
    return 3;
}

/* -------------------------------------------------------------------------
 * Test 2: NMC_EXEC (EMBEDDING_LOOKUP) -> ACK
 * ---------------------------------------------------------------------- */
static int test_nmc_exec(edgecoh_transport_t *transport) {
    printf("Test 2: NMC_EXEC (EMBEDDING_LOOKUP) -> ACK ... ");
    fflush(stdout);

    edgecoh_nmc_exec_msg_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.msg_type    = EDGECOH_MSG_NMC_EXEC;
    msg.header.flags       = 0;
    msg.header.tensor_id   = 2;
    msg.header.payload_len = (uint32_t)(sizeof(edgecoh_nmc_exec_msg_t)
                                        - sizeof(edgecoh_header_t));
    msg.nmc_op             = EDGECOH_NMC_EMBEDDING_LOOKUP;
    msg.table_base_addr    = 0x00100000;   /* FPGA DDR2 base for embedding table */
    msg.table_rows         = 1000;
    msg.table_cols         = 64;
    msg.input_addr         = 0x00500000;   /* indices stored here */
    msg.input_len          = 1;            /* single lookup */
    msg.output_addr        = 0x00600000;   /* write result here */

    uint8_t tx_buf[sizeof(edgecoh_nmc_exec_msg_t)];
    int n = edgecoh_serialize(&msg, tx_buf, sizeof(tx_buf));
    if (n < 0) {
        printf("FAIL (serialize error)\n");
        return 1;
    }

    int sent = edgecoh_transport_send(transport, tx_buf, n);
    if (sent != n) {
        printf("FAIL (send error: sent=%d expected=%d)\n", sent, n);
        return 1;
    }

    int rc = wait_ack(transport, /*expected_tensor_id=*/2, ACK_TIMEOUT_MS);
    if (rc == 0)  { printf("PASS\n"); return 0; }
    if (rc == -1) { printf("FAIL (timeout)\n"); return 2; }
    printf("FAIL (unexpected response)\n");
    return 3;
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */
int main(int argc, char *argv[]) {
    (void)argc; (void)argv;

    printf("=== EdgeCoh USB Loopback Integration Test ===\n");
    printf("Target: Nexys 4 DDR (VID=0x%04X, PID=0x%04X)\n", NEXYS_VID, NEXYS_PID);
    printf("\n");

    edgecoh_transport_t *transport = edgecoh_transport_open(NEXYS_VID, NEXYS_PID);
    if (!transport) {
        fprintf(stderr, "ERROR: Could not open USB transport.\n");
        fprintf(stderr, "  Ensure the Nexys 4 DDR is connected (VID=0x%04X PID=0x%04X)\n",
                NEXYS_VID, NEXYS_PID);
        fprintf(stderr, "  and programmed with the SplitInfer bitstream.\n");
        return 1;
    }
    printf("Transport opened successfully.\n\n");

    int failures = 0;
    failures += (test_sync_barrier(transport) != 0) ? 1 : 0;
    failures += (test_nmc_exec(transport)     != 0) ? 1 : 0;

    edgecoh_transport_close(transport);

    printf("\n");
    if (failures == 0) {
        printf("=== ALL TESTS PASSED ===\n");
    } else {
        printf("=== %d TEST(S) FAILED ===\n", failures);
    }

    return failures > 0 ? 1 : 0;
}
