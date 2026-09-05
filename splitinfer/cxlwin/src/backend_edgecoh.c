/* Real-FPGA backend: page faults become EdgeCoh DATA_READ / DATA_WRITE. */
#define _GNU_SOURCE
#include "cxlwin/cxlwin.h"
#include <edgecoh/messages.h>
#include <edgecoh/transport.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHUNK 4096
#define TIMEOUT_MS 20000

typedef struct { edgecoh_transport_t *t; int owns; } ec_t;

static int recv_exact(edgecoh_transport_t *t, uint8_t *dst, size_t len) {
    size_t got = 0;
    while (got < len) {
        int want = (int)((len - got) > CHUNK ? CHUNK : (len - got));
        int n = edgecoh_transport_recv(t, dst + got, want, TIMEOUT_MS);
        if (n <= 0) return -1;
        got += (size_t)n;
    }
    return 0;
}

static int ec_read(void *ctx, uint32_t a, void *dst, size_t len) {
    ec_t *e = ctx;
    edgecoh_data_read_msg_t m; memset(&m, 0, sizeof(m));
    m.header.msg_type = EDGECOH_MSG_DATA_READ;
    m.header.payload_len = 8;
    m.ddr2_addr = a; m.read_len = (uint32_t)len;
    uint8_t buf[32];
    int n = edgecoh_serialize(&m, buf, sizeof(buf));
    if (n < 0 || edgecoh_transport_send(e->t, buf, n) != n) return -1;
    edgecoh_header_t h;
    if (edgecoh_recv_header(e->t, &h, TIMEOUT_MS) < 0) return -1;
    if (h.msg_type != EDGECOH_MSG_DATA_RESPONSE) return -1;
    size_t take = h.payload_len < len ? h.payload_len : len;
    if (recv_exact(e->t, dst, take) != 0) return -1;
    if (h.payload_len > take) {          /* drain */
        uint8_t junk[256]; size_t rem = h.payload_len - take;
        while (rem) { int w = (int)(rem > sizeof(junk) ? sizeof(junk) : rem);
                      int g = edgecoh_transport_recv(e->t, junk, w, TIMEOUT_MS); if (g <= 0) break; rem -= (size_t)g; }
    }
    return take == len ? 0 : -1;
}

static int ec_write(void *ctx, uint32_t a, const void *src, size_t len) {
    ec_t *e = ctx;
    const uint8_t *p = src;
    while (len) {
        size_t chunk = len > CHUNK ? CHUNK : len;
        edgecoh_data_write_msg_t m; memset(&m, 0, sizeof(m));
        m.header.msg_type = EDGECOH_MSG_DATA_WRITE;
        m.header.payload_len = (uint32_t)(4 + chunk);
        m.ddr2_addr = a;
        uint8_t *buf = malloc(sizeof(m) + chunk);
        int n = edgecoh_serialize(&m, buf, (int)sizeof(m));
        if (n < 0) { free(buf); return -1; }
        memcpy(buf + n, p, chunk);
        int rc = edgecoh_transport_send(e->t, buf, n + (int)chunk);
        free(buf);
        if (rc != n + (int)chunk) return -1;
        edgecoh_header_t h;
        if (edgecoh_recv_header(e->t, &h, TIMEOUT_MS) < 0 || h.msg_type == EDGECOH_MSG_ERROR) return -1;
        p += chunk; a += (uint32_t)chunk; len -= chunk;
    }
    return 0;
}

static void ec_destroy(void *ctx) {
    ec_t *e = ctx;
    if (e->owns) edgecoh_transport_close(e->t);
    free(e);
}

int cxlwin_backend_edgecoh_create(cxlwin_backend_t *out, struct edgecoh_transport *t, int owns) {
    ec_t *e = calloc(1, sizeof(*e));
    if (!e) return -1;
    e->t = t; e->owns = owns;
    out->ctx = e; out->read = ec_read; out->write = ec_write; out->destroy = ec_destroy;
    return 0;
}
