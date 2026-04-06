/* splitinfer/protocol/include/edgecoh/transport.h */
#ifndef EDGECOH_TRANSPORT_H
#define EDGECOH_TRANSPORT_H

#include <stdint.h>
#include "messages.h"

typedef struct edgecoh_transport edgecoh_transport_t;

/* Nexys 4 DDR FTDI: VID=0x0403, PID=0x6010 (FT2232HQ dual channel). */
edgecoh_transport_t *edgecoh_transport_open(uint16_t vid, uint16_t pid);
void edgecoh_transport_close(edgecoh_transport_t *t);
int edgecoh_transport_send(edgecoh_transport_t *t, const uint8_t *buf, int len);
int edgecoh_transport_recv(edgecoh_transport_t *t, uint8_t *buf, int buf_len, int timeout_ms);
int edgecoh_send_msg(edgecoh_transport_t *t, const void *msg);
int edgecoh_recv_header(edgecoh_transport_t *t, edgecoh_header_t *hdr, int timeout_ms);

#endif /* EDGECOH_TRANSPORT_H */
