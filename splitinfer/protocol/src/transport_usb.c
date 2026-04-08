/* splitinfer/protocol/src/transport_usb.c
 *
 * EdgeCoh USB transport over FTDI FT2232HQ Channel B (UART mode).
 *
 * Implementation strategy: use the kernel's ftdi_sio driver via /dev/ttyUSB1
 * (Channel B).  This avoids the complexity of FTDI baud-rate encoding,
 * bitmode setup, modem-status header stripping, and bulk endpoint addressing
 * — the kernel driver handles all of it.
 *
 * The Nexys 4 DDR's FPGA UART (usb_interface.v) is hardcoded to 921600 8N1.
 * The kernel ftdi_sio driver supports custom baud rates via the standard
 * termios cfsetspeed() interface.
 *
 * Channel A (/dev/ttyUSB0) is the JTAG/programming channel — we never touch it.
 * Channel B (/dev/ttyUSB1) is wired to the FPGA's uart_rx/uart_tx pins.
 */
#include "edgecoh/transport.h"
#include "edgecoh/messages.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <time.h>

/* Default tty path for FT2232HQ Channel B on the Nexys 4 DDR.
 * In the rare case the system enumerates differently, the user can set
 * EDGECOH_TTY=/dev/ttyUSB2 (etc.) in the environment. */
#define EDGECOH_DEFAULT_TTY "/dev/ttyUSB1"

struct edgecoh_transport {
    int fd;
};

/* Open the tty, configure 921600 8N1 raw mode. */
edgecoh_transport_t *edgecoh_transport_open(uint16_t vid, uint16_t pid) {
    (void)vid; (void)pid;  /* identification handled by udev → /dev/ttyUSB1 */

    const char *path = getenv("EDGECOH_TTY");
    if (!path || !*path) path = EDGECOH_DEFAULT_TTY;

    int fd = open(path, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        fprintf(stderr, "edgecoh: cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }

    /* Drop O_NONBLOCK after open — we use select() for timeouts. */
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);

    struct termios tio;
    if (tcgetattr(fd, &tio) != 0) {
        fprintf(stderr, "edgecoh: tcgetattr failed: %s\n", strerror(errno));
        close(fd);
        return NULL;
    }

    /* Raw mode: no echo, no line processing, 8N1, no flow control. */
    cfmakeraw(&tio);
    tio.c_cflag |= (CLOCAL | CREAD);
    tio.c_cflag &= ~CRTSCTS;          /* no hardware flow control */
    tio.c_cflag &= ~PARENB;           /* no parity */
    tio.c_cflag &= ~CSTOPB;           /* 1 stop bit */
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |= CS8;               /* 8 data bits */
    tio.c_iflag &= ~(IXON | IXOFF | IXANY); /* no software flow control */
    tio.c_cc[VMIN]  = 0;              /* select() handles blocking */
    tio.c_cc[VTIME] = 0;

    /* Set baud rate to 115200 — FPGA's usb_interface.v BAUD_RATE parameter
     * was lowered from 921600 → 115200 for robust UART framing margin
     * (see fpga/src/usb_interface.v for rationale). */
    if (cfsetispeed(&tio, B115200) != 0 ||
        cfsetospeed(&tio, B115200) != 0) {
        fprintf(stderr, "edgecoh: cfsetspeed B115200 failed: %s\n",
                strerror(errno));
        close(fd);
        return NULL;
    }

    if (tcsetattr(fd, TCSANOW, &tio) != 0) {
        fprintf(stderr, "edgecoh: tcsetattr failed: %s\n", strerror(errno));
        close(fd);
        return NULL;
    }

    /* Flush any stale bytes from previous sessions.
     * tcflush() clears kernel buffers; we also drain the FT2232HQ on-chip
     * FIFO by actively reading with a short timeout until it's empty.
     * Without this, any leftover bytes from a previous process (e.g. a
     * partial ACK that wasn't consumed by a previous failed run) will
     * cause the next receive to misframe. */
    tcflush(fd, TCIOFLUSH);
    {
        uint8_t junk[256];
        fd_set rfds;
        struct timeval tv;
        for (int attempt = 0; attempt < 20; attempt++) {
            FD_ZERO(&rfds); FD_SET(fd, &rfds);
            tv.tv_sec = 0; tv.tv_usec = 10000; /* 10 ms */
            if (select(fd + 1, &rfds, NULL, NULL, &tv) <= 0) break;
            if (read(fd, junk, sizeof(junk)) <= 0) break;
        }
        tcflush(fd, TCIOFLUSH); /* final sweep */
    }

    edgecoh_transport_t *t = calloc(1, sizeof(*t));
    if (!t) { close(fd); return NULL; }
    t->fd = fd;
    return t;
}

void edgecoh_transport_close(edgecoh_transport_t *t) {
    if (!t) return;
    if (t->fd >= 0) close(t->fd);
    free(t);
}

int edgecoh_transport_send(edgecoh_transport_t *t, const uint8_t *buf, int len) {
    if (!t || t->fd < 0 || !buf || len <= 0) return -1;

    int total = 0;
    while (total < len) {
        ssize_t n = write(t->fd, buf + total, (size_t)(len - total));
        if (n < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "edgecoh: write error: %s\n", strerror(errno));
            return -1;
        }
        total += (int)n;
    }

    /* Drain TX so the bytes are actually on the wire before we return. */
    tcdrain(t->fd);
    return total;
}

int edgecoh_transport_recv(edgecoh_transport_t *t, uint8_t *buf, int buf_len,
                           int timeout_ms) {
    if (!t || t->fd < 0 || !buf || buf_len <= 0) return -1;

    /* Compute absolute deadline so partial reads don't restart the timeout. */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long deadline_ms = (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000
                            + (long long)timeout_ms;

    int total = 0;
    while (total < buf_len) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        long long now_ms = (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
        long long remaining = deadline_ms - now_ms;
        if (remaining <= 0) return total;

        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(t->fd, &rfds);
        struct timeval tv = {
            .tv_sec  = remaining / 1000,
            .tv_usec = (remaining % 1000) * 1000,
        };

        int rc = select(t->fd + 1, &rfds, NULL, NULL, &tv);
        if (rc < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "edgecoh: select error: %s\n", strerror(errno));
            return -1;
        }
        if (rc == 0) return total;  /* timeout */

        ssize_t n = read(t->fd, buf + total, (size_t)(buf_len - total));
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            fprintf(stderr, "edgecoh: read error: %s\n", strerror(errno));
            return -1;
        }
        if (n == 0) return total;  /* EOF — shouldn't happen on tty */
        total += (int)n;
    }
    return total;
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
