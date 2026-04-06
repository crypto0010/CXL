/* splitinfer/protocol/src/messages.c */
#include "edgecoh/messages.h"
#include <string.h>

static int msg_total_size(uint8_t msg_type) {
    switch (msg_type) {
    case EDGECOH_MSG_TRANSFER_OWNERSHIP: return (int)sizeof(edgecoh_transfer_msg_t);
    case EDGECOH_MSG_PREFETCH: return (int)sizeof(edgecoh_prefetch_msg_t);
    case EDGECOH_MSG_SYNC_BARRIER:
    case EDGECOH_MSG_ACK:
    case EDGECOH_MSG_ERROR: return (int)sizeof(edgecoh_header_t);
    case EDGECOH_MSG_DATA_WRITE: return (int)sizeof(edgecoh_data_write_msg_t);
    case EDGECOH_MSG_DATA_READ: return (int)sizeof(edgecoh_data_read_msg_t);
    case EDGECOH_MSG_NMC_EXEC: return (int)sizeof(edgecoh_nmc_exec_msg_t);
    case EDGECOH_MSG_NMC_DONE:
    case EDGECOH_MSG_DATA_RESPONSE: return (int)sizeof(edgecoh_header_t);
    default: return -1;
    }
}

int edgecoh_serialize(const void *msg, uint8_t *buf, int buf_len) {
    const edgecoh_header_t *hdr = (const edgecoh_header_t *)msg;
    int size = msg_total_size(hdr->msg_type);
    if (size < 0 || size > buf_len) return -1;
    memcpy(buf, msg, size);
    return size;
}

int edgecoh_deserialize_header(const uint8_t *buf, int buf_len, edgecoh_header_t *out) {
    if (buf_len < (int)sizeof(edgecoh_header_t)) return -1;
    memcpy(out, buf, sizeof(edgecoh_header_t));
    return (int)out->msg_type;
}
