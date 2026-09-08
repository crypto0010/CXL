/* DDR2 data-integrity test over the live link (board or emulator).
 *   1. lane-order probe: write bytes 0..15 at a 16-aligned address, read the
 *      word back as one 16-byte read and as 16 single-byte reads.
 *   2. unaligned 4,096-byte pseudo-random round trip with guard bytes.
 *   3. 64 KiB sequential round trip (chunked 4 KiB writes, one 64 KiB read).
 * Usage: test_dma_integrity [emul]
 */
#include "edgecoh/transport.h"
#include "edgecoh/transport_emul.h"
#include "edgecoh/messages.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
#define TO 20000

static int wr(edgecoh_transport_t *t, uint32_t a, const void *src, size_t len) {
    const uint8_t *p = src;
    while (len) {
        size_t chunk = len > 4096 ? 4096 : len;
        edgecoh_data_write_msg_t m = {0}; m.header.msg_type = EDGECOH_MSG_DATA_WRITE; m.header.payload_len = 4 + chunk; m.ddr2_addr = a;
        uint8_t *b = malloc(sizeof(m) + chunk); int n = edgecoh_serialize(&m, b, sizeof(m)); memcpy(b + n, p, chunk);
        int rc = edgecoh_transport_send(t, b, n + (int)chunk); free(b); if (rc != n + (int)chunk) return -1;
        edgecoh_header_t h; if (edgecoh_recv_header(t, &h, TO) < 0 || h.msg_type != EDGECOH_MSG_ACK) return -2;
        p += chunk; a += chunk; len -= chunk;
    }
    return 0;
}
static int rd(edgecoh_transport_t *t, uint32_t a, void *dst, size_t len) {
    edgecoh_data_read_msg_t m = {0}; m.header.msg_type = EDGECOH_MSG_DATA_READ; m.header.payload_len = 8; m.ddr2_addr = a; m.read_len = len;
    uint8_t b[32]; int n = edgecoh_serialize(&m, b, sizeof(b)); if (edgecoh_transport_send(t, b, n) != n) return -1;
    edgecoh_header_t h; if (edgecoh_recv_header(t, &h, TO) < 0) return -2; if (h.msg_type != EDGECOH_MSG_DATA_RESPONSE) return -3;
    size_t got = 0; while (got < len) { int r = edgecoh_transport_recv(t, (uint8_t *)dst + got, (int)(len - got > 4096 ? 4096 : len - got), TO); if (r <= 0) return -4; got += r; }
    return 0;
}
static uint8_t prng(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return (uint8_t)(*s >> 24); }

int main(int argc, char **argv) {
    edgecoh_transport_t *t;
    if (argc > 1 && !strcmp(argv[1], "emul")) { edgecoh_emul_params_t p = { .ddr2_bytes = 1 << 20 }; t = edgecoh_transport_open_emul(&p); }
    else t = edgecoh_transport_open(0x0403, 0x6010);
    if (!t) { printf("transport open failed\n"); return 1; }

    /* 1. lane order */
    uint8_t seq[16], w16[16], b1[16]; for (int i = 0; i < 16; i++) seq[i] = 0x10 + i;
    CHECK(wr(t, 0x40000, seq, 16) == 0, "lane: write");
    CHECK(rd(t, 0x40000, w16, 16) == 0, "lane: 16B read");
    for (int i = 0; i < 16; i++) { uint8_t v = 0; CHECK(rd(t, 0x40000 + i, &v, 1) == 0, "lane: 1B read %d", i); b1[i] = v; }
    printf("  wrote  : "); for (int i = 0; i < 16; i++) printf("%02x ", seq[i]); printf("\n");
    printf("  read16 : "); for (int i = 0; i < 16; i++) printf("%02x ", w16[i]); printf("\n");
    printf("  read1x : "); for (int i = 0; i < 16; i++) printf("%02x ", b1[i]); printf("\n");
    CHECK(memcmp(seq, w16, 16) == 0, "lane order (16B read)");
    CHECK(memcmp(seq, b1, 16) == 0, "lane order (byte reads)");

    /* 2. unaligned 4096 with guards */
    uint32_t base = 0x50003; uint8_t g = 0xA5;
    for (int i = 1; i <= 20; i++) { wr(t, base - i, &g, 1); wr(t, base + 4096 - 1 + i, &g, 1); }
    uint8_t *pat = malloc(4096), *rb = malloc(4096); uint32_t s = 12345; for (int i = 0; i < 4096; i++) pat[i] = prng(&s);
    CHECK(wr(t, base, pat, 4096) == 0, "4K: write"); CHECK(rd(t, base, rb, 4096) == 0, "4K: read");
    int bad = 0, first = -1; for (int i = 0; i < 4096; i++) if (rb[i] != pat[i]) { bad++; if (first < 0) first = i; }
    CHECK(bad == 0, "4K unaligned round trip: %d bad bytes, first at %d (got %02x want %02x)", bad, first, first >= 0 ? rb[first] : 0, first >= 0 ? pat[first] : 0);
    int gbad = 0; for (int i = 1; i <= 20; i++) { uint8_t v; rd(t, base - i, &v, 1); if (v != g) gbad++; rd(t, base + 4096 - 1 + i, &v, 1); if (v != g) gbad++; }
    CHECK(gbad == 0, "guard bytes clobbered: %d", gbad);

    /* 3. 64 KiB */
    size_t N = 65536; uint8_t *big = malloc(N), *bb = malloc(N); s = 777; for (size_t i = 0; i < N; i++) big[i] = prng(&s);
    CHECK(wr(t, 0x80000, big, N) == 0, "64K: write"); CHECK(rd(t, 0x80000, bb, N) == 0, "64K: read");
    bad = 0; first = -1; for (size_t i = 0; i < N; i++) if (bb[i] != big[i]) { bad++; if (first < 0) first = (int)i; }
    CHECK(bad == 0, "64K round trip: %d bad bytes, first at %d", bad, first);
    if (bad) { int shown = 0; for (size_t i = 0; i < N && shown < 6; i++) if (bb[i] != big[i]) { printf("    [%zu] got %02x want %02x\n", i, bb[i], big[i]); shown++; } }
    edgecoh_transport_close(t);
    printf(fails ? "FAILED (%d)\n" : "ALL PASSED\n", fails); return fails != 0;
}
