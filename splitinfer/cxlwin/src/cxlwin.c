/* splitinfer/cxlwin/src/cxlwin.c — see cxlwin.h for the model. */
#define _GNU_SOURCE
#include "cxlwin/cxlwin.h"

#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define MAX_WINDOWS 8

struct cxlwin {
    uint8_t         *base;
    size_t           size;
    size_t           npages;
    uint32_t         dev_base;
    unsigned         prefetch;
    int              strict;
    cxlwin_backend_t be;
    uint8_t         *state;        /* cxlwin_state_t per page */
    pthread_mutex_t  lock;         /* serialises state changes + backend I/O */
    cxlwin_stats_t   stats;
};

/* ── Global fault router (signal handlers have no context argument) ───── */
static cxlwin_t        *g_windows[MAX_WINDOWS];
static pthread_mutex_t  g_reg_lock = PTHREAD_MUTEX_INITIALIZER;
static int              g_pipe[2] = {-1, -1};
static sem_t            g_done;
static pthread_t        g_service;
static int              g_installed = 0;
static struct sigaction g_prev_sa;

static size_t PAGE;

static uint64_t now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static cxlwin_t *find_window(const void *addr) {
    for (int i = 0; i < MAX_WINDOWS; i++) {
        cxlwin_t *w = g_windows[i];
        if (w && (const uint8_t *)addr >= w->base && (const uint8_t *)addr < w->base + w->size)
            return w;
    }
    return NULL;
}

/* ── Page operations (called with w->lock held, from the service thread) ─ */

static int fetch_pages(cxlwin_t *w, size_t first, size_t count) {
    /* Trim to pages that actually need fetching (INVALID / DEVICE). */
    while (count > 0 && w->state[first] != CXLWIN_INVALID && w->state[first] != CXLWIN_DEVICE) {
        first++; count--;
    }
    while (count > 0 && w->state[first + count - 1] != CXLWIN_INVALID &&
           w->state[first + count - 1] != CXLWIN_DEVICE)
        count--;
    if (count == 0) return 0;
    for (size_t p = first; p < first + count; p++) w->state[p] = CXLWIN_FETCHING;

    uint8_t *dst = w->base + first * PAGE;
    size_t len = count * PAGE;
    if (mprotect(dst, len, PROT_READ | PROT_WRITE) != 0) return -1;
    uint64_t t0 = now_ns();
    int rc = w->be.read(w->be.ctx, w->dev_base + (uint32_t)(first * PAGE), dst, len);
    uint64_t dt = now_ns() - t0;
    w->stats.fetch_calls++;
    w->stats.fetch_ns_total += dt;
    if (dt > w->stats.fetch_ns_max) w->stats.fetch_ns_max = dt;
    if (rc != 0) {
        mprotect(dst, len, PROT_NONE);
        for (size_t p = first; p < first + count; p++) w->state[p] = CXLWIN_INVALID;
        return -1;
    }
    w->stats.pages_fetched += count;
    w->stats.bytes_in += len;
    if (mprotect(dst, len, PROT_READ) != 0) return -1;
    for (size_t p = first; p < first + count; p++) w->state[p] = CXLWIN_SHARED;
    return 0;
}

static int flush_page(cxlwin_t *w, size_t p) {
    if (w->state[p] != CXLWIN_MODIFIED) return 0;
    w->state[p] = CXLWIN_FLUSHING;
    uint8_t *src = w->base + p * PAGE;
    uint64_t t0 = now_ns();
    int rc = w->be.write(w->be.ctx, w->dev_base + (uint32_t)(p * PAGE), src, PAGE);
    w->stats.flush_calls++;
    w->stats.flush_ns_total += now_ns() - t0;
    if (rc != 0) { w->state[p] = CXLWIN_MODIFIED; return -1; }
    w->stats.pages_flushed++;
    w->stats.bytes_out += PAGE;
    mprotect(src, PAGE, PROT_READ);
    w->state[p] = CXLWIN_SHARED;
    return 0;
}

/* Service one fault at `addr`. */
static void service_fault(cxlwin_t *w, uint8_t *addr) {
    size_t p = (size_t)(addr - w->base) / PAGE;
    pthread_mutex_lock(&w->lock);
    switch (w->state[p]) {
    case CXLWIN_DEVICE:
        w->stats.bias_violations++;
        if (w->strict) {
            fprintf(stderr, "cxlwin: STRICT: host access to DEVICE-bias page %zu\n", p);
            abort();
        }
        /* fall through: service as INVALID */
    case CXLWIN_INVALID: {
        size_t count = w->prefetch ? w->prefetch : 1;
        if (p + count > w->npages) count = w->npages - p;
        w->stats.read_faults++;
        fetch_pages(w, p, count);
        break;
    }
    case CXLWIN_SHARED:
        /* Second fault on a readable page == a store: upgrade. */
        w->stats.write_upgrades++;
        mprotect(w->base + p * PAGE, PAGE, PROT_READ | PROT_WRITE);
        w->state[p] = CXLWIN_MODIFIED;
        break;
    case CXLWIN_MODIFIED:
        /* Spurious (e.g. raced with a flush); make sure it's writable. */
        mprotect(w->base + p * PAGE, PAGE, PROT_READ | PROT_WRITE);
        break;
    case CXLWIN_FETCHING:
    case CXLWIN_FLUSHING:
        /* Cannot happen: transient states only exist while this thread holds
         * the lock inside fetch/flush. */
        break;
    }
    pthread_mutex_unlock(&w->lock);
}

static void *service_main(void *arg) {
    (void)arg;
    void *addr;
    for (;;) {
        ssize_t n = read(g_pipe[0], &addr, sizeof(addr));
        if (n != (ssize_t)sizeof(addr)) { if (n <= 0 && errno != EINTR) break; continue; }
        if (addr == NULL) break;                    /* shutdown */
        pthread_mutex_lock(&g_reg_lock);
        cxlwin_t *w = find_window(addr);
        pthread_mutex_unlock(&g_reg_lock);
        if (w) service_fault(w, (uint8_t *)addr);
        sem_post(&g_done);
    }
    return NULL;
}

static void segv_handler(int sig, siginfo_t *si, void *uc) {
    void *addr = si->si_addr;
    /* Only async-signal-safe calls below. */
    if (!find_window(addr)) {
        /* Not ours: restore previous disposition and return; the instruction
         * re-executes and the default action (or previous handler) fires. */
        sigaction(SIGSEGV, &g_prev_sa, NULL);
        return;
    }
    ssize_t r = write(g_pipe[1], &addr, sizeof(addr));
    (void)r; (void)sig; (void)uc;
    while (sem_wait(&g_done) != 0 && errno == EINTR) {}
}

static int ensure_installed(void) {
    if (g_installed) return 0;
    PAGE = (size_t)sysconf(_SC_PAGESIZE);
    if (pipe(g_pipe) != 0) return -1;
    if (sem_init(&g_done, 0, 0) != 0) return -1;
    if (pthread_create(&g_service, NULL, service_main, NULL) != 0) return -1;
    struct sigaction sa; memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = segv_handler;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;   /* NODEFER: a nested fault in the
                                                service thread must be deliverable */
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGSEGV, &sa, &g_prev_sa) != 0) return -1;
    g_installed = 1;
    return 0;
}

/* ── Public API ──────────────────────────────────────────────────────── */

size_t cxlwin_page_size(void) { return PAGE ? PAGE : (size_t)sysconf(_SC_PAGESIZE); }

cxlwin_t *cxlwin_create(const cxlwin_config_t *cfg, const cxlwin_backend_t *backend) {
    if (!cfg || !backend || !backend->read || !backend->write) { errno = EINVAL; return NULL; }
    if (ensure_installed() != 0) return NULL;
    cxlwin_t *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->npages = (cfg->size + PAGE - 1) / PAGE;
    w->size = w->npages * PAGE;
    w->dev_base = cfg->dev_base;
    w->prefetch = cfg->prefetch_pages ? cfg->prefetch_pages : 1;
    w->strict = cfg->strict;
    w->be = *backend;
    w->state = calloc(w->npages, 1);
    pthread_mutex_init(&w->lock, NULL);
    w->base = mmap(NULL, w->size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (w->base == MAP_FAILED || !w->state) { free(w->state); free(w); return NULL; }
    pthread_mutex_lock(&g_reg_lock);
    int slot = -1;
    for (int i = 0; i < MAX_WINDOWS; i++) if (!g_windows[i]) { slot = i; break; }
    if (slot >= 0) g_windows[slot] = w;
    pthread_mutex_unlock(&g_reg_lock);
    if (slot < 0) { munmap(w->base, w->size); free(w->state); free(w); errno = ENOSPC; return NULL; }
    return w;
}

void cxlwin_destroy(cxlwin_t *w) {
    if (!w) return;
    pthread_mutex_lock(&g_reg_lock);
    for (int i = 0; i < MAX_WINDOWS; i++) if (g_windows[i] == w) g_windows[i] = NULL;
    pthread_mutex_unlock(&g_reg_lock);
    munmap(w->base, w->size);
    if (w->be.destroy) w->be.destroy(w->be.ctx);
    pthread_mutex_destroy(&w->lock);
    free(w->state);
    free(w);
}

void  *cxlwin_base(const cxlwin_t *w) { return w->base; }
size_t cxlwin_size(const cxlwin_t *w) { return w->size; }

static int range_pages(const cxlwin_t *w, size_t off, size_t len, size_t *first, size_t *last) {
    if (off >= w->size) return -1;
    if (len == 0 || off + len > w->size) len = w->size - off;
    *first = off / PAGE;
    *last  = (off + len - 1) / PAGE;
    return 0;
}

int cxlwin_release(cxlwin_t *w, size_t off, size_t len) {
    size_t a, b; if (range_pages(w, off, len, &a, &b)) return -1;
    int rc = 0;
    pthread_mutex_lock(&w->lock);
    for (size_t p = a; p <= b; p++) {
        if (w->state[p] == CXLWIN_MODIFIED && flush_page(w, p) != 0) rc = -1;
        mprotect(w->base + p * PAGE, PAGE, PROT_NONE);
        w->state[p] = CXLWIN_DEVICE;
    }
    pthread_mutex_unlock(&w->lock);
    return rc;
}

int cxlwin_acquire(cxlwin_t *w, size_t off, size_t len) {
    size_t a, b; if (range_pages(w, off, len, &a, &b)) return -1;
    pthread_mutex_lock(&w->lock);
    for (size_t p = a; p <= b; p++)
        if (w->state[p] == CXLWIN_DEVICE) w->state[p] = CXLWIN_INVALID;
    pthread_mutex_unlock(&w->lock);
    return 0;
}

int cxlwin_flush(cxlwin_t *w, size_t off, size_t len) {
    size_t a, b; if (range_pages(w, off, len, &a, &b)) return -1;
    int rc = 0;
    pthread_mutex_lock(&w->lock);
    for (size_t p = a; p <= b; p++) if (flush_page(w, p) != 0) rc = -1;
    pthread_mutex_unlock(&w->lock);
    return rc;
}

int cxlwin_prefetch(cxlwin_t *w, size_t off, size_t len) {
    size_t a, b; if (range_pages(w, off, len, &a, &b)) return -1;
    pthread_mutex_lock(&w->lock);
    /* Fetch maximal runs of INVALID pages so each run is one backend call. */
    int rc = 0;
    size_t p = a;
    while (p <= b) {
        while (p <= b && w->state[p] != CXLWIN_INVALID) p++;
        if (p > b) break;
        size_t q = p;
        while (q <= b && w->state[q] == CXLWIN_INVALID) q++;
        if (fetch_pages(w, p, q - p) != 0) rc = -1;
        p = q;
    }
    pthread_mutex_unlock(&w->lock);
    return rc;
}

int cxlwin_invalidate(cxlwin_t *w, size_t off, size_t len) {
    size_t a, b; if (range_pages(w, off, len, &a, &b)) return -1;
    pthread_mutex_lock(&w->lock);
    for (size_t p = a; p <= b; p++) {
        if (w->state[p] == CXLWIN_DEVICE) continue;
        mprotect(w->base + p * PAGE, PAGE, PROT_NONE);
        w->state[p] = CXLWIN_INVALID;
    }
    pthread_mutex_unlock(&w->lock);
    return 0;
}

cxlwin_state_t cxlwin_page_state(const cxlwin_t *w, size_t off) {
    if (off >= w->size) return CXLWIN_INVALID;
    return (cxlwin_state_t)w->state[off / PAGE];
}

const char *cxlwin_state_name(cxlwin_state_t s) {
    switch (s) {
    case CXLWIN_INVALID:  return "INVALID";
    case CXLWIN_FETCHING: return "FETCHING";
    case CXLWIN_SHARED:   return "SHARED";
    case CXLWIN_MODIFIED: return "MODIFIED";
    case CXLWIN_FLUSHING: return "FLUSHING";
    case CXLWIN_DEVICE:   return "DEVICE";
    }
    return "?";
}

void cxlwin_get_stats(const cxlwin_t *w, cxlwin_stats_t *out) { *out = w->stats; }
void cxlwin_reset_stats(cxlwin_t *w) { memset(&w->stats, 0, sizeof(w->stats)); }
