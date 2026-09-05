/* splitinfer/cxlwin/include/cxlwin/cxlwin.h
 *
 * cxlwin — a host load/store window onto device-attached memory.
 *
 * Maps a region of FPGA DDR2 into the host virtual address space.  The host
 * dereferences ordinary pointers; page faults are serviced by EdgeCoh
 * transfers.  Coherence is software-managed at page granularity with an
 * explicit bias model (host bias / device bias) in the style of CXL.mem
 * Type 2 devices — stated as an analogy at page rather than cache-line
 * granularity, not as equivalence.
 *
 * Mechanism: SIGSEGV + mprotect (classic software DSM: Ivy, TreadMarks).
 * userfaultfd is not compiled into the Tegra kernel, and this needs no
 * kernel feature, so it ports to any edge SoC.
 *
 * Page states — with TRANSIENT states that track outstanding responses:
 *
 *   INVALID   PROT_NONE   no valid host copy; device holds the data
 *   FETCHING  PROT_NONE   transient: a DATA_READ is outstanding
 *   SHARED    PROT_READ   host has a clean copy; device copy still valid
 *   MODIFIED  PROT_RW     host copy dirty; device copy stale
 *   FLUSHING  PROT_READ   transient: a DATA_WRITE is outstanding
 *   DEVICE    PROT_NONE   device bias: an NMC engine owns the page
 *
 * Transitions:
 *   load  / INVALID  -> FETCHING -> SHARED
 *   store / SHARED   -> MODIFIED
 *   store / INVALID  -> FETCHING -> SHARED -> (refault) -> MODIFIED
 *   release()        MODIFIED -> FLUSHING -> DEVICE ; SHARED|INVALID -> DEVICE
 *   acquire()        DEVICE -> INVALID   (refilled on demand)
 *   flush()          MODIFIED -> FLUSHING -> SHARED
 *   prefetch()       INVALID  -> FETCHING -> SHARED   (batched, one RTT)
 *
 * A host access to a DEVICE-bias page is a coherence violation.  By default
 * it is counted and serviced transparently (so experiments still complete);
 * CXLWIN_STRICT aborts instead.
 *
 * Thread model: the faulting thread blocks in the signal handler on a
 * semaphore while a dedicated service thread performs the transfer and the
 * mprotect.  The handler itself only calls async-signal-safe functions.
 */
#ifndef CXLWIN_H
#define CXLWIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cxlwin cxlwin_t;

/* Device memory access.  Both return 0 on success.  `dev_addr` is a byte
 * address in FPGA DDR2.  Implementations must be safe to call from the
 * service thread only (they are never called from a signal handler). */
typedef struct {
    void *ctx;
    int (*read)(void *ctx, uint32_t dev_addr, void *dst, size_t len);
    int (*write)(void *ctx, uint32_t dev_addr, const void *src, size_t len);
    void (*destroy)(void *ctx);
} cxlwin_backend_t;

typedef enum {
    CXLWIN_INVALID = 0,
    CXLWIN_FETCHING,
    CXLWIN_SHARED,
    CXLWIN_MODIFIED,
    CXLWIN_FLUSHING,
    CXLWIN_DEVICE,
} cxlwin_state_t;

typedef struct {
    uint64_t read_faults;        /* faults serviced with a fetch            */
    uint64_t write_upgrades;     /* SHARED -> MODIFIED faults (no transfer)  */
    uint64_t pages_fetched;
    uint64_t pages_flushed;
    uint64_t bytes_in;           /* device -> host                          */
    uint64_t bytes_out;          /* host -> device                          */
    uint64_t fetch_ns_total;     /* time spent inside backend read()        */
    uint64_t fetch_ns_max;
    uint64_t flush_ns_total;
    uint64_t fetch_calls;        /* backend read() invocations (== RTTs)    */
    uint64_t flush_calls;
    uint64_t bias_violations;    /* host touched a DEVICE-bias page         */
} cxlwin_stats_t;

typedef struct {
    size_t   size;               /* window bytes; rounded up to page size   */
    uint32_t dev_base;           /* DDR2 byte address of window offset 0    */
    unsigned prefetch_pages;     /* pages fetched per miss (>=1)            */
    int      strict;             /* abort on bias violation                 */
} cxlwin_config_t;

cxlwin_t *cxlwin_create(const cxlwin_config_t *cfg, const cxlwin_backend_t *backend);
void      cxlwin_destroy(cxlwin_t *w);

void  *cxlwin_base(const cxlwin_t *w);
size_t cxlwin_size(const cxlwin_t *w);
size_t cxlwin_page_size(void);

/* Ownership / bias transitions over [off, off+len).  Return 0 on success. */
int cxlwin_release(cxlwin_t *w, size_t off, size_t len);   /* host -> device bias */
int cxlwin_acquire(cxlwin_t *w, size_t off, size_t len);   /* device -> host bias */
int cxlwin_flush(cxlwin_t *w, size_t off, size_t len);     /* write back, keep host copy */
int cxlwin_prefetch(cxlwin_t *w, size_t off, size_t len);  /* fetch now, one batched RTT per call */
int cxlwin_invalidate(cxlwin_t *w, size_t off, size_t len);/* drop host copies without writeback */

cxlwin_state_t cxlwin_page_state(const cxlwin_t *w, size_t off);
const char    *cxlwin_state_name(cxlwin_state_t s);

void cxlwin_get_stats(const cxlwin_t *w, cxlwin_stats_t *out);
void cxlwin_reset_stats(cxlwin_t *w);

/* ── Backends ─────────────────────────────────────────────────────────── */

/* In-process device emulator: a byte array with a modelled link.
 * bandwidth_bytes_per_s == 0 and rtt_ns == 0 disable the delay model. */
int   cxlwin_backend_emul_create(cxlwin_backend_t *out, size_t dev_bytes,
                                 double bandwidth_bytes_per_s, uint64_t rtt_ns);
void *cxlwin_backend_emul_memory(const cxlwin_backend_t *be);   /* direct access for tests */

/* Real FPGA via the EdgeCoh transport (DATA_READ / DATA_WRITE). */
struct edgecoh_transport;
int cxlwin_backend_edgecoh_create(cxlwin_backend_t *out, struct edgecoh_transport *t,
                                  int owns_transport);

#ifdef __cplusplus
}
#endif
#endif /* CXLWIN_H */
