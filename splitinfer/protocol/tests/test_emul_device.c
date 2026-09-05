/* Virtual-FPGA tests through the public transport API: the same message
 * bytes the runtime sends to the board.  Mirrors fpga/sim/v2 golden tests. */
#include "edgecoh/transport_emul.h"
#include "edgecoh/messages.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static int xfer_write(edgecoh_transport_t *t, uint32_t a, const void *src, size_t len) {
    edgecoh_data_write_msg_t m = {0}; m.header.msg_type = EDGECOH_MSG_DATA_WRITE; m.header.payload_len = 4 + len; m.ddr2_addr = a;
    uint8_t *b = malloc(sizeof(m) + len); int n = edgecoh_serialize(&m, b, sizeof(m)); memcpy(b + n, src, len);
    int rc = edgecoh_transport_send(t, b, n + (int)len); free(b);
    edgecoh_header_t h; if (rc < 0 || edgecoh_recv_header(t, &h, 100) < 0 || h.msg_type != EDGECOH_MSG_ACK) return -1;
    return 0;
}
static int xfer_read(edgecoh_transport_t *t, uint32_t a, void *dst, size_t len) {
    edgecoh_data_read_msg_t m = {0}; m.header.msg_type = EDGECOH_MSG_DATA_READ; m.header.payload_len = 8; m.ddr2_addr = a; m.read_len = len;
    uint8_t b[32]; int n = edgecoh_serialize(&m, b, sizeof(b)); edgecoh_transport_send(t, b, n);
    edgecoh_header_t h; if (edgecoh_recv_header(t, &h, 100) < 0 || h.msg_type != EDGECOH_MSG_DATA_RESPONSE || h.payload_len != len) return -1;
    return edgecoh_transport_recv(t, dst, (int)len, 100) == (int)len ? 0 : -1;
}
static int nmc(edgecoh_transport_t *t, uint8_t op, uint32_t tb, uint32_t rows, uint32_t cols, uint32_t ia, uint32_t il, uint32_t oa) {
    edgecoh_nmc_exec_msg_t m = {0}; m.header.msg_type = EDGECOH_MSG_NMC_EXEC; m.header.payload_len = 25;
    m.nmc_op = op; m.table_base_addr = tb; m.table_rows = rows; m.table_cols = cols; m.input_addr = ia; m.input_len = il; m.output_addr = oa;
    uint8_t b[64]; int n = edgecoh_serialize(&m, b, sizeof(b)); edgecoh_transport_send(t, b, n);
    edgecoh_header_t h; return (edgecoh_recv_header(t, &h, 100) >= 0 && h.msg_type == EDGECOH_MSG_ACK) ? 0 : -1;
}

int main(void) {
    edgecoh_emul_params_t p = { .ddr2_bytes = 1 << 20 };
    edgecoh_transport_t *t = edgecoh_transport_open_emul(&p);
    /* 1. data round trip */
    uint8_t w[5000], r[5000]; for (int i = 0; i < 5000; i++) w[i] = (uint8_t)(i * 7 + 3);
    CHECK(xfer_write(t, 1003, w, 5000) == 0, "write"); CHECK(xfer_read(t, 1003, r, 5000) == 0, "read");
    CHECK(memcmp(w, r, 5000) == 0, "data round trip");
    /* 2. INT8 FC vs reference */
    enum { M = 24, K = 48 }; int8_t W[M * K], X[K]; int32_t Y[M], Yref[M];
    for (int i = 0; i < M * K; i++) W[i] = (int8_t)rand(); for (int k = 0; k < K; k++) X[k] = (int8_t)rand();
    for (int m = 0; m < M; m++) { Yref[m] = 0; for (int k = 0; k < K; k++) Yref[m] += W[m * K + k] * X[k]; }
    xfer_write(t, 0x1000, W, sizeof W); xfer_write(t, 0x3000, X, sizeof X);
    CHECK(nmc(t, 0x02, 0x1000, M, K, 0x3000, 0, 0x4000) == 0, "nmc fc");
    xfer_read(t, 0x4000, Y, sizeof Y);
    CHECK(memcmp(Y, Yref, sizeof Y) == 0, "INT8 FC matches reference");
    /* 3. epilogue vs reference (mult 1187, shift 14, relu) */
    enum { N = 44 }; int32_t A[N], B[N]; int8_t O[48], Oref[48] = {0};
    for (int i = 0; i < N; i++) { A[i] = rand() >> 6; B[i] = rand() >> 12; int64_t s = ((int64_t)A[i] + B[i]) * 1187 >> 14; if (s < 0) s = 0; Oref[i] = s > 127 ? 127 : (int8_t)s; }
    xfer_write(t, 0x5000, A, sizeof A); xfer_write(t, 0x6000, B, sizeof B);
    CHECK(nmc(t, 0x03, 4 | ((0x80 | 14) << 8), (N + 3) / 4, 0x6000, 0x5000, 1187, 0x7000) == 0, "nmc epilogue");
    xfer_read(t, 0x7000, O, 48);
    CHECK(memcmp(O, Oref, 48) == 0, "epilogue matches reference (incl. zero pad)");
    /* 4. embedding gather */
    uint8_t T[32 * 32]; for (int i = 0; i < 1024; i++) T[i] = (uint8_t)rand(); uint32_t IDX[12]; for (int i = 0; i < 12; i++) IDX[i] = rand() % 32;
    xfer_write(t, 0x8000, T, sizeof T); xfer_write(t, 0x9000, IDX, sizeof IDX);
    CHECK(nmc(t, 0x01, 0x8000, 32, 32, 0x9000, 12, 0xA000) == 0, "nmc embedding");
    uint8_t G[12 * 32]; xfer_read(t, 0xA000, G, sizeof G);
    int errs = 0; for (int i = 0; i < 12; i++) if (memcmp(G + i * 32, T + IDX[i] * 32, 32)) errs++;
    CHECK(errs == 0, "embedding gather: %d wrong rows", errs);
    edgecoh_emul_stats_t s; edgecoh_emul_get_stats(t, &s);
    printf("msgs=%llu nmc=%llu engine_cycles=%llu\n", (unsigned long long)s.msgs, (unsigned long long)s.nmc_exec, (unsigned long long)s.engine_cycles);
    edgecoh_transport_close(t);
    printf(fails ? "FAILED (%d)\n" : "ALL PASSED\n", fails); return fails != 0;
}
