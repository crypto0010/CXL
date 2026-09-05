/* cxlwin unit tests against the emulator backend.
 * Every test exercises the ORDINARY load/store path — no explicit copies. */
#include "cxlwin/cxlwin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static cxlwin_t *mk(size_t size, unsigned prefetch, cxlwin_backend_t *be_out) {
    cxlwin_backend_t be;
    if (cxlwin_backend_emul_create(&be, size, 0, 0) != 0) return NULL;
    uint8_t *dev = cxlwin_backend_emul_memory(&be);
    for (size_t i = 0; i < size; i++) dev[i] = (uint8_t)(i * 13 + 5);
    cxlwin_config_t cfg = { .size = size, .dev_base = 0, .prefetch_pages = prefetch, .strict = 0 };
    if (be_out) *be_out = be;
    return cxlwin_create(&cfg, &be);
}

static void test_demand_read(void) {
    printf("demand read faults\n");
    size_t P = cxlwin_page_size();
    cxlwin_t *w = mk(8 * P, 1, NULL);
    volatile uint8_t *m = cxlwin_base(w);
    long sum = 0; int errs = 0;
    for (size_t i = 0; i < 8 * P; i += 101) { uint8_t v = m[i]; if (v != (uint8_t)(i * 13 + 5)) errs++; sum += v; }
    cxlwin_stats_t s; cxlwin_get_stats(w, &s);
    CHECK(errs == 0, "%d mismatches", errs);
    CHECK(s.read_faults == 8, "read_faults=%llu", (unsigned long long)s.read_faults);
    CHECK(s.pages_fetched == 8, "pages_fetched=%llu", (unsigned long long)s.pages_fetched);
    CHECK(s.fetch_calls == 8, "fetch_calls=%llu", (unsigned long long)s.fetch_calls);
    CHECK(cxlwin_page_state(w, 0) == CXLWIN_SHARED, "state %s", cxlwin_state_name(cxlwin_page_state(w, 0)));
    /* second pass: no new faults */
    for (size_t i = 0; i < 8 * P; i += 101) sum += m[i];
    cxlwin_get_stats(w, &s);
    CHECK(s.read_faults == 8, "refault: read_faults=%llu", (unsigned long long)s.read_faults);
    cxlwin_destroy(w);
}

static void test_prefetch_batches_rtts(void) {
    printf("prefetch batching\n");
    size_t P = cxlwin_page_size();
    cxlwin_t *w = mk(64 * P, 16, NULL);
    volatile uint8_t *m = cxlwin_base(w);
    long sum = 0;
    for (size_t i = 0; i < 64 * P; i += P) sum += m[i];
    cxlwin_stats_t s; cxlwin_get_stats(w, &s);
    CHECK(s.pages_fetched == 64, "pages_fetched=%llu", (unsigned long long)s.pages_fetched);
    CHECK(s.fetch_calls == 4, "fetch_calls=%llu (expected 4 batches of 16)", (unsigned long long)s.fetch_calls);
    cxlwin_destroy(w);
}

static void test_write_upgrade_and_release(void) {
    printf("store -> MODIFIED, release -> writeback -> DEVICE\n");
    size_t P = cxlwin_page_size();
    cxlwin_backend_t be;
    cxlwin_t *w = mk(4 * P, 1, &be);
    uint8_t *dev = cxlwin_backend_emul_memory(&be);
    volatile uint8_t *m = cxlwin_base(w);
    m[P + 7] = 0xAB;                             /* store to an INVALID page */
    CHECK(cxlwin_page_state(w, P) == CXLWIN_MODIFIED, "state after store: %s", cxlwin_state_name(cxlwin_page_state(w, P)));
    CHECK(dev[P + 7] != 0xAB, "device must NOT see the store before release");
    cxlwin_stats_t s; cxlwin_get_stats(w, &s);
    CHECK(s.read_faults == 1 && s.write_upgrades == 1, "faults r=%llu w=%llu", (unsigned long long)s.read_faults, (unsigned long long)s.write_upgrades);
    CHECK(cxlwin_release(w, P, P) == 0, "release");
    CHECK(dev[P + 7] == 0xAB, "device sees the store after release");
    CHECK(cxlwin_page_state(w, P) == CXLWIN_DEVICE, "state after release: %s", cxlwin_state_name(cxlwin_page_state(w, P)));
    cxlwin_get_stats(w, &s);
    CHECK(s.pages_flushed == 1 && s.bytes_out == P, "flushed=%llu bytes_out=%llu", (unsigned long long)s.pages_flushed, (unsigned long long)s.bytes_out);
    /* the rest of the page round-tripped intact */
    int errs = 0; for (size_t i = 0; i < P; i++) if (i != 7 && dev[P + i] != (uint8_t)((P + i) * 13 + 5)) errs++;
    CHECK(errs == 0, "%d bytes of the page corrupted by writeback", errs);
    cxlwin_destroy(w);
}

static void test_device_bias_then_acquire(void) {
    printf("device modifies while DEVICE-bias; acquire refetches\n");
    size_t P = cxlwin_page_size();
    cxlwin_backend_t be;
    cxlwin_t *w = mk(2 * P, 1, &be);
    uint8_t *dev = cxlwin_backend_emul_memory(&be);
    volatile uint8_t *m = cxlwin_base(w);
    uint8_t before = m[3];                       /* SHARED */
    cxlwin_release(w, 0, P);                     /* DEVICE */
    dev[3] = (uint8_t)(before + 1);              /* "NMC engine" writes */
    cxlwin_stats_t s; cxlwin_reset_stats(w);
    uint8_t stale = m[3];                        /* violation: touched DEVICE page */
    cxlwin_get_stats(w, &s);
    CHECK(s.bias_violations == 1, "bias_violations=%llu", (unsigned long long)s.bias_violations);
    CHECK(stale == (uint8_t)(before + 1), "violation is serviced with fresh data (non-strict)");
    cxlwin_release(w, 0, P);
    dev[3] = (uint8_t)(before + 2);
    cxlwin_acquire(w, 0, P);                     /* INVALID -> demand refetch */
    CHECK(cxlwin_page_state(w, 0) == CXLWIN_INVALID, "state after acquire: %s", cxlwin_state_name(cxlwin_page_state(w, 0)));
    CHECK(m[3] == (uint8_t)(before + 2), "acquire sees device write: %u vs %u", m[3], (unsigned)(before + 2));
    cxlwin_destroy(w);
}

static void test_flush_keeps_copy(void) {
    printf("flush writes back and keeps SHARED\n");
    size_t P = cxlwin_page_size();
    cxlwin_backend_t be;
    cxlwin_t *w = mk(2 * P, 1, &be);
    uint8_t *dev = cxlwin_backend_emul_memory(&be);
    volatile uint8_t *m = cxlwin_base(w);
    m[10] = 0x5C;
    cxlwin_flush(w, 0, P);
    CHECK(dev[10] == 0x5C, "device sees flushed byte");
    CHECK(cxlwin_page_state(w, 0) == CXLWIN_SHARED, "state after flush: %s", cxlwin_state_name(cxlwin_page_state(w, 0)));
    uint8_t again = m[10];
    cxlwin_stats_t s; cxlwin_get_stats(w, &s);
    CHECK(again == 0x5C && s.read_faults == 1, "no refetch after flush");
    m[11] = 0x11;                                /* re-upgrade */
    CHECK(cxlwin_page_state(w, 0) == CXLWIN_MODIFIED, "re-upgrade after flush");
    cxlwin_destroy(w);
}

static void test_large_sequential_memcpy(void) {
    printf("memcpy through the window (libc, no per-byte loop)\n");
    size_t P = cxlwin_page_size();
    size_t N = 256 * P;
    cxlwin_backend_t be;
    cxlwin_t *w = mk(N, 32, &be);
    uint8_t *dev = cxlwin_backend_emul_memory(&be);
    uint8_t *buf = malloc(N);
    memcpy(buf, cxlwin_base(w), N);              /* ordinary memcpy faults its way through */
    CHECK(memcmp(buf, dev, N) == 0, "memcpy content mismatch");
    cxlwin_stats_t s; cxlwin_get_stats(w, &s);
    CHECK(s.fetch_calls == 8, "fetch_calls=%llu (256 pages / 32)", (unsigned long long)s.fetch_calls);
    free(buf); cxlwin_destroy(w);
}

int main(void) {
    test_demand_read();
    test_prefetch_batches_rtts();
    test_write_upgrade_and_release();
    test_device_bias_then_acquire();
    test_flush_keeps_copy();
    test_large_sequential_memcpy();
    printf(fails ? "FAILED (%d)\n" : "ALL PASSED\n", fails);
    return fails != 0;
}
