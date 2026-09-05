/* splitinfer/protocol/src/transport_emul.c — virtual FPGA.  See header. */
#define _GNU_SOURCE
#include "edgecoh/transport_priv.h"
#include "edgecoh/transport_emul.h"
#include "edgecoh/messages.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    edgecoh_emul_params_t p;
    uint8_t *ddr2;
    /* RX assembly (host -> device) */
    uint8_t  rx[1 << 16];  size_t rx_len;
    /* TX queue (device -> host) */
    uint8_t *tx; size_t tx_len, tx_cap, tx_head;
    edgecoh_emul_stats_t st;
} emul_t;

#define E(t) ((emul_t *)(t)->priv)

static void sleep_s(double s) {
    if (s <= 0) return;
    struct timespec ts = { (time_t)s, (long)((s - (double)(time_t)s) * 1e9) };
    nanosleep(&ts, NULL);
}
static void link_delay(emul_t *e, size_t nbytes, int msgs) {
    double s = e->p.link_rtt_s * msgs;
    if (e->p.link_bytes_per_s > 0) s += (double)nbytes / e->p.link_bytes_per_s;
    e->st.emulated_link_ns += (uint64_t)(s * 1e9);
    sleep_s(s);
}
static void compute_delay(emul_t *e, double cycles) {
    e->st.engine_cycles += (uint64_t)cycles;
    if (e->p.engine_clock_hz <= 0) return;
    double s = cycles / e->p.engine_clock_hz;
    e->st.emulated_compute_ns += (uint64_t)(s * 1e9);
    sleep_s(s);
}

static void tx_push(emul_t *e, const void *buf, size_t n) {
    if (e->tx_len + n > e->tx_cap) {
        e->tx_cap = (e->tx_len + n) * 2 + 4096;
        e->tx = realloc(e->tx, e->tx_cap);
    }
    memcpy(e->tx + e->tx_len, buf, n); e->tx_len += n;
}
static void tx_ack(emul_t *e, uint16_t tensor_id) {
    edgecoh_header_t h = { EDGECOH_MSG_ACK, 0, tensor_id, 0 };
    tx_push(e, &h, sizeof(h));
}

static inline uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static inline int32_t  rds32(const uint8_t *p) { return (int32_t)rd32(p); }
static inline void     wr32(uint8_t *p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }
static inline int8_t   sat8(int64_t x) { return x > 127 ? 127 : x < -128 ? -128 : (int8_t)x; }

/* ── Engine models (exact RTL semantics) ─────────────────────────────── */

static int in_range(emul_t *e, uint64_t a, uint64_t n) { return a + n <= e->p.ddr2_bytes; }

static void eng_embedding(emul_t *e, uint32_t table, uint32_t dim, uint32_t idx_addr, uint32_t n, uint32_t out) {
    /* dim in BYTES; bursts_per_embed = dim >> 4 (RTL truncates to whole 16-byte words). */
    uint32_t bytes = (dim >> 4) << 4;
    for (uint32_t i = 0; i < n; i++) {
        if (!in_range(e, (uint64_t)idx_addr + i * 4 + 4, 0)) return;
        uint32_t ix = rd32(e->ddr2 + idx_addr + i * 4);
        uint64_t src = (uint64_t)table + (uint64_t)ix * dim, dst = (uint64_t)out + (uint64_t)i * dim;
        if (!in_range(e, src, bytes) || !in_range(e, dst, bytes)) return;
        memcpy(e->ddr2 + dst, e->ddr2 + src, bytes);
    }
    /* 1 index read + (dim/16) row reads + (dim/16) writes per index, ~26 cycles per DDR2 op */
    compute_delay(e, (double)n * (1 + 2.0 * (dim >> 4)) * 26.0);
}

static void eng_int8_fc(emul_t *e, uint32_t w, uint32_t x, uint32_t y, uint32_t M, uint32_t K) {
    /* out[m] = sum_k W[m*K+k] * X[k], INT32; M%8==0, K%16==0 as the RTL requires. */
    if (!in_range(e, w, (uint64_t)M * K) || !in_range(e, x, K) || !in_range(e, y, (uint64_t)M * 4)) return;
    for (uint32_t m = 0; m < M; m++) {
        int32_t acc = 0;
        const int8_t *wr = (const int8_t *)(e->ddr2 + w + (uint64_t)m * K);
        const int8_t *xr = (const int8_t *)(e->ddr2 + x);
        for (uint32_t k = 0; k < K; k++) acc += (int32_t)wr[k] * (int32_t)xr[k];
        wr32(e->ddr2 + y + m * 4, (uint32_t)acc);
    }
    double mpc = e->p.mac_per_cycle > 0 ? e->p.mac_per_cycle : 0.684;
    compute_delay(e, (double)M * K / mpc);
}

static void eng_elementwise(emul_t *e, uint32_t base_field, uint32_t nwords, uint32_t addr2,
                            uint32_t in, uint32_t in_len_field, uint32_t out) {
    uint32_t op = base_field & 7, scale = (base_field >> 8) & 0xFF;
    int16_t mult = (int16_t)(in_len_field & 0xFFFF);
    uint32_t shift = scale & 31; int relu = (scale >> 7) & 1;
    if (op == 4) {
        /* EPILOGUE: every 4 input words (16 INT32) -> one 16-byte INT8 word.  Partial tail zero-padded. */
        uint32_t nout = (nwords + 3) / 4;
        if (!in_range(e, in, (uint64_t)nwords * 16) || !in_range(e, addr2, (uint64_t)nwords * 16) || !in_range(e, out, (uint64_t)nout * 16)) return;
        memset(e->ddr2 + out, 0, (size_t)nout * 16);
        for (uint32_t i = 0; i < nwords * 4; i++) {
            int64_t s = (int64_t)rds32(e->ddr2 + in + i * 4) + (int64_t)rds32(e->ddr2 + addr2 + i * 4);
            int64_t p = s * (int64_t)mult;
            int64_t sh = p >> shift;               /* arithmetic shift, matches >>> */
            if (relu && sh < 0) sh = 0;
            e->ddr2[out + i] = (uint8_t)sat8(sh);
        }
        compute_delay(e, (double)nwords * 60.0);
        return;
    }
    if (!in_range(e, in, (uint64_t)nwords * 16) || !in_range(e, out, (uint64_t)nwords * 16)) return;
    if (op == 3 && !in_range(e, addr2, (uint64_t)nwords * 16)) return;
    for (uint32_t i = 0; i < nwords * 16; i++) {
        int8_t v = (int8_t)e->ddr2[in + i]; int16_t r;
        switch (op) {
        case 0: r = v < 0 ? 0 : v; break;
        case 1: r = (int16_t)v + (int16_t)scale; break;
        case 2: r = (int16_t)v * (int16_t)scale; break;
        case 3: r = (int16_t)v + (int16_t)(int8_t)e->ddr2[addr2 + i]; break;
        default: r = v; break;
        }
        e->ddr2[out + i] = (uint8_t)sat8(r);
    }
    compute_delay(e, (double)nwords * (op == 3 ? 60.0 : 32.0));
}

/* ── Message processing ──────────────────────────────────────────────── */

static void process(emul_t *e, const uint8_t *m, size_t n) {
    edgecoh_header_t h;
    if (edgecoh_deserialize_header(m, (int)n, &h) < 0) return;
    const uint8_t *pl = m + sizeof(h);
    e->st.msgs++;
    link_delay(e, n, 1);
    switch (h.msg_type) {
    case EDGECOH_MSG_DATA_WRITE: {
        uint32_t a = rd32(pl); size_t len = h.payload_len - 4;
        if (in_range(e, a, len)) memcpy(e->ddr2 + a, pl + 4, len);
        e->st.data_write_bytes += len;
        tx_ack(e, h.tensor_id);
        link_delay(e, sizeof(h), 0);
        break;
    }
    case EDGECOH_MSG_DATA_READ: {
        uint32_t a = rd32(pl), len = rd32(pl + 4);
        edgecoh_header_t r = { EDGECOH_MSG_DATA_RESPONSE, 0, h.tensor_id, len };
        tx_push(e, &r, sizeof(r));
        if (in_range(e, a, len)) tx_push(e, e->ddr2 + a, len);
        else { uint8_t z[256] = {0}; for (uint32_t i = 0; i < len; i += 256) tx_push(e, z, len - i > 256 ? 256 : len - i); }
        e->st.data_read_bytes += len;
        link_delay(e, sizeof(r) + len, 0);
        break;
    }
    case EDGECOH_MSG_NMC_EXEC: {
        uint8_t op = pl[0];
        uint32_t tb = rd32(pl + 1), rows = rd32(pl + 5), cols = rd32(pl + 9);
        uint32_t ia = rd32(pl + 13), il = rd32(pl + 17), oa = rd32(pl + 21);
        e->st.nmc_exec++;
        switch (op) {
        case 0x01: eng_embedding(e, tb, cols, ia, il, oa); break;
        case 0x02: eng_int8_fc(e, tb, ia, oa, rows, cols); break;
        case 0x03: eng_elementwise(e, tb, rows, cols, ia, il, oa); break;
        default: break;
        }
        tx_ack(e, h.tensor_id);
        link_delay(e, sizeof(h), 0);
        break;
    }
    default:   /* SYNC_BARRIER, TRANSFER_OWNERSHIP, PREFETCH: ACK like the controller */
        tx_ack(e, h.tensor_id);
        link_delay(e, sizeof(h), 0);
        break;
    }
}

static int emul_send(edgecoh_transport_t *t, const uint8_t *buf, int len) {
    emul_t *e = E(t);
    if (len <= 0) return -1;
    if (e->rx_len + (size_t)len > sizeof(e->rx)) { e->rx_len = 0; return -1; }
    memcpy(e->rx + e->rx_len, buf, (size_t)len); e->rx_len += (size_t)len;
    /* Consume every complete message in the buffer (a send may carry several). */
    for (;;) {
        if (e->rx_len < sizeof(edgecoh_header_t)) break;
        edgecoh_header_t h;
        if (edgecoh_deserialize_header(e->rx, (int)e->rx_len, &h) < 0) { e->rx_len = 0; break; }
        size_t need = sizeof(h) + h.payload_len;
        if (e->rx_len < need) break;
        process(e, e->rx, need);
        memmove(e->rx, e->rx + need, e->rx_len - need); e->rx_len -= need;
    }
    return len;
}

static int emul_recv(edgecoh_transport_t *t, uint8_t *buf, int buf_len, int timeout_ms) {
    (void)timeout_ms;
    emul_t *e = E(t);
    size_t avail = e->tx_len - e->tx_head;
    size_t n = avail < (size_t)buf_len ? avail : (size_t)buf_len;
    memcpy(buf, e->tx + e->tx_head, n); e->tx_head += n;
    if (e->tx_head == e->tx_len) { e->tx_head = e->tx_len = 0; }
    return (int)n;
}

static void emul_close(edgecoh_transport_t *t) {
    emul_t *e = E(t); free(e->ddr2); free(e->tx); free(e); free(t);
}

static const edgecoh_transport_ops_t emul_ops = { emul_send, emul_recv, emul_close };

edgecoh_transport_t *edgecoh_transport_open_emul(const edgecoh_emul_params_t *p) {
    emul_t *e = calloc(1, sizeof(*e));
    edgecoh_transport_t *t = calloc(1, sizeof(*t));
    if (!e || !t) { free(e); free(t); return NULL; }
    if (p) e->p = *p;
    if (e->p.ddr2_bytes == 0) e->p.ddr2_bytes = 128u << 20;
    e->ddr2 = calloc(e->p.ddr2_bytes, 1);
    if (!e->ddr2) { free(e); free(t); return NULL; }
    t->ops = &emul_ops; t->priv = e;
    return t;
}

uint8_t *edgecoh_emul_ddr2(edgecoh_transport_t *t, size_t *size_out) {
    if (size_out) *size_out = E(t)->p.ddr2_bytes;
    return E(t)->ddr2;
}
void edgecoh_emul_get_stats(edgecoh_transport_t *t, edgecoh_emul_stats_t *out) { *out = E(t)->st; }
