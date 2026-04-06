/* splitinfer/protocol/src/transport_usb.c */
#include "edgecoh/transport.h"
#include "edgecoh/messages.h"
#include <libusb-1.0/libusb.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define FTDI_INTERFACE 1
#define FTDI_EP_OUT    0x02
#define FTDI_EP_IN     0x81

struct edgecoh_transport {
    libusb_context       *ctx;
    libusb_device_handle *dev;
};

edgecoh_transport_t *edgecoh_transport_open(uint16_t vid, uint16_t pid) {
    edgecoh_transport_t *t = calloc(1, sizeof(*t));
    if (!t) return NULL;

    if (libusb_init(&t->ctx) != 0) {
        free(t);
        return NULL;
    }

    t->dev = libusb_open_device_with_vid_pid(t->ctx, vid, pid);
    if (!t->dev) {
        fprintf(stderr, "edgecoh: cannot open USB device %04x:%04x\n", vid, pid);
        libusb_exit(t->ctx);
        free(t);
        return NULL;
    }

    libusb_detach_kernel_driver(t->dev, FTDI_INTERFACE);
    if (libusb_claim_interface(t->dev, FTDI_INTERFACE) != 0) {
        fprintf(stderr, "edgecoh: cannot claim interface %d\n", FTDI_INTERFACE);
        libusb_close(t->dev);
        libusb_exit(t->ctx);
        free(t);
        return NULL;
    }

    return t;
}

void edgecoh_transport_close(edgecoh_transport_t *t) {
    if (!t) return;
    libusb_release_interface(t->dev, FTDI_INTERFACE);
    libusb_close(t->dev);
    libusb_exit(t->ctx);
    free(t);
}

int edgecoh_transport_send(edgecoh_transport_t *t, const uint8_t *buf, int len) {
    int transferred = 0;
    int rc = libusb_bulk_transfer(t->dev, FTDI_EP_OUT,
                                  (uint8_t *)buf, len, &transferred, 5000);
    if (rc != 0) {
        fprintf(stderr, "edgecoh: USB send error: %s\n", libusb_error_name(rc));
        return -1;
    }
    return transferred;
}

int edgecoh_transport_recv(edgecoh_transport_t *t, uint8_t *buf, int buf_len,
                           int timeout_ms) {
    int transferred = 0;
    int rc = libusb_bulk_transfer(t->dev, FTDI_EP_IN,
                                  buf, buf_len, &transferred, timeout_ms);
    if (rc != 0 && rc != LIBUSB_ERROR_TIMEOUT) {
        fprintf(stderr, "edgecoh: USB recv error: %s\n", libusb_error_name(rc));
        return -1;
    }
    return transferred;
}

int edgecoh_send_msg(edgecoh_transport_t *t, const void *msg) {
    uint8_t buf[512];
    int len = edgecoh_serialize(msg, buf, sizeof(buf));
    if (len < 0) return -1;
    int sent = edgecoh_transport_send(t, buf, len);
    if (sent != len) return -1;
    return 0;
}

int edgecoh_recv_header(edgecoh_transport_t *t, edgecoh_header_t *hdr,
                        int timeout_ms) {
    uint8_t buf[sizeof(edgecoh_header_t)];
    int n = edgecoh_transport_recv(t, buf, sizeof(buf), timeout_ms);
    if (n < (int)sizeof(edgecoh_header_t)) return -1;
    return edgecoh_deserialize_header(buf, n, hdr);
}
