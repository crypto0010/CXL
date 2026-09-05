/* Private transport vtable.  Public API in transport.h dispatches through it
 * so the USB transport and the in-process device emulator are
 * interchangeable above this line. */
#ifndef EDGECOH_TRANSPORT_PRIV_H
#define EDGECOH_TRANSPORT_PRIV_H
#include "edgecoh/transport.h"

typedef struct {
    int  (*send)(edgecoh_transport_t *t, const uint8_t *buf, int len);
    int  (*recv)(edgecoh_transport_t *t, uint8_t *buf, int buf_len, int timeout_ms);
    void (*close)(edgecoh_transport_t *t);
} edgecoh_transport_ops_t;

struct edgecoh_transport {
    const edgecoh_transport_ops_t *ops;
    void *priv;
};
#endif
