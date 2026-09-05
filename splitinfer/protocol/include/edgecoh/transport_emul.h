/* splitinfer/protocol/include/edgecoh/transport_emul.h
 *
 * Virtual FPGA: an EdgeCoh transport whose far end is an in-process model of
 * the Nexys 4 DDR design — 128 MB DDR2 as a byte array and the three NMC
 * engines with the SAME integer semantics as the v2 RTL (validated against
 * the same golden vectors).  Everything above the transport API is
 * byte-for-byte identical between the emulator and the board.
 *
 * Timing is EMULATED: link bytes/s + per-message RTT, and engine cycle
 * counts derived from simulation, are converted to wall-clock sleeps so
 * that pool-vs-NMC experiments can be run off-board.  Results obtained this
 * way are labelled EMULATED in the paper and never mixed with MEASURED.
 */
#ifndef EDGECOH_TRANSPORT_EMUL_H
#define EDGECOH_TRANSPORT_EMUL_H
#include "edgecoh/transport.h"
#include <stddef.h>

typedef struct {
    size_t   ddr2_bytes;          /* default 128 MiB */
    double   link_bytes_per_s;    /* 0 = no delay */
    double   link_rtt_s;          /* per message, 0 = none */
    double   engine_clock_hz;     /* 0 = no compute delay; RTL: 81.25e6 */
    double   mac_per_cycle;       /* from sim: 0.684 */
} edgecoh_emul_params_t;

edgecoh_transport_t *edgecoh_transport_open_emul(const edgecoh_emul_params_t *p);
/* Direct DDR2 access for tests / reference checks. */
uint8_t *edgecoh_emul_ddr2(edgecoh_transport_t *t, size_t *size_out);
/* Counters. */
typedef struct {
    uint64_t msgs, data_write_bytes, data_read_bytes, nmc_exec;
    uint64_t emulated_link_ns, emulated_compute_ns, engine_cycles;
} edgecoh_emul_stats_t;
void edgecoh_emul_get_stats(edgecoh_transport_t *t, edgecoh_emul_stats_t *out);
#endif
