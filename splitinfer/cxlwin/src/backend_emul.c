/* In-process device emulator backend: a byte array plus a link delay model. */
#define _GNU_SOURCE
#include "cxlwin/cxlwin.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint8_t *mem;
    size_t   size;
    double   bw;        /* bytes per second, 0 = infinite */
    uint64_t rtt_ns;    /* per-call latency */
} emul_t;

static void delay(const emul_t *e, size_t len) {
    uint64_t ns = e->rtt_ns;
    if (e->bw > 0) ns += (uint64_t)((double)len / e->bw * 1e9);
    if (ns == 0) return;
    struct timespec ts = { (time_t)(ns / 1000000000ull), (long)(ns % 1000000000ull) };
    nanosleep(&ts, NULL);
}

static int emul_read(void *ctx, uint32_t a, void *dst, size_t len) {
    emul_t *e = ctx;
    if ((size_t)a + len > e->size) return -1;
    delay(e, len);
    memcpy(dst, e->mem + a, len);
    return 0;
}
static int emul_write(void *ctx, uint32_t a, const void *src, size_t len) {
    emul_t *e = ctx;
    if ((size_t)a + len > e->size) return -1;
    delay(e, len);
    memcpy(e->mem + a, src, len);
    return 0;
}
static void emul_destroy(void *ctx) { emul_t *e = ctx; free(e->mem); free(e); }

int cxlwin_backend_emul_create(cxlwin_backend_t *out, size_t dev_bytes,
                               double bandwidth_bytes_per_s, uint64_t rtt_ns) {
    emul_t *e = calloc(1, sizeof(*e));
    if (!e) return -1;
    e->mem = calloc(dev_bytes, 1);
    if (!e->mem) { free(e); return -1; }
    e->size = dev_bytes; e->bw = bandwidth_bytes_per_s; e->rtt_ns = rtt_ns;
    out->ctx = e; out->read = emul_read; out->write = emul_write; out->destroy = emul_destroy;
    return 0;
}

void *cxlwin_backend_emul_memory(const cxlwin_backend_t *be) {
    return ((emul_t *)be->ctx)->mem;
}
