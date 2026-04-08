/* splitinfer/protocol/include/edgecoh/messages.h */
#ifndef EDGECOH_MESSAGES_H
#define EDGECOH_MESSAGES_H

#include <stdint.h>

/* EdgeCoh protocol message types — inspired by CXL.mem
 *
 * Completion semantics:
 *   The current FPGA firmware (edgecoh_controller.v) uniformly emits
 *   EDGECOH_MSG_ACK (0xFE) for the completion of every dispatched
 *   message — SYNC_BARRIER, NMC_EXEC, DATA_WRITE, etc.  The host is
 *   expected to know which message it just sent and interpret the
 *   ACK accordingly.
 *
 *   EDGECOH_MSG_NMC_DONE (0x21) is reserved for future firmware that
 *   wants to distinguish "NMC operation complete" from "generic ACK"
 *   — for example, to deliver NMC-specific status flags or error codes
 *   in the same response.  Today's firmware does NOT emit this opcode.
 *   The runtime accepts both 0xFE and 0x21 as completion indicators
 *   for forward compatibility.  See runtime/src/fpga_executor.cpp
 *   FpgaExecutor::execute() for the dual-opcode acceptance check.
 *
 *   EDGECOH_MSG_DATA_RESPONSE (0x12) is the one exception: it IS
 *   distinct from ACK because it carries a payload (the requested
 *   tensor bytes), so the host must know to read additional bytes
 *   beyond the 8-byte header.
 */

typedef enum {
    EDGECOH_MSG_TRANSFER_OWNERSHIP = 0x01,
    EDGECOH_MSG_PREFETCH           = 0x02,
    EDGECOH_MSG_SYNC_BARRIER       = 0x03,
    EDGECOH_MSG_DATA_WRITE         = 0x10,  /* Host -> FPGA: write tensor data */
    EDGECOH_MSG_DATA_READ          = 0x11,  /* Host -> FPGA: request tensor data */
    EDGECOH_MSG_DATA_RESPONSE      = 0x12,  /* FPGA -> Host: tensor data response */
    EDGECOH_MSG_NMC_EXEC           = 0x20,  /* Host -> FPGA: execute NMC operation */
    EDGECOH_MSG_NMC_DONE           = 0x21,  /* RESERVED — see header comment.
                                              * Current firmware uses MSG_ACK
                                              * for all completions.  Runtime
                                              * accepts both for fwd compat. */
    EDGECOH_MSG_ACK                = 0xFE,  /* Universal completion response */
    EDGECOH_MSG_ERROR              = 0xFF,
} edgecoh_msg_type_t;

typedef enum {
    EDGECOH_DEV_HOST = 0,   /* Jetson */
    EDGECOH_DEV_FPGA = 1,
} edgecoh_device_t;

typedef enum {
    EDGECOH_NMC_EMBEDDING_LOOKUP = 0x01,
    EDGECOH_NMC_INT8_FC          = 0x02,
    EDGECOH_NMC_RELU             = 0x03,
    EDGECOH_NMC_QUANT_ADD        = 0x04,
    EDGECOH_NMC_SCALE            = 0x05,
} edgecoh_nmc_op_t;

/* All messages share a common 8-byte header */
typedef struct __attribute__((packed)) {
    uint8_t  msg_type;       /* edgecoh_msg_type_t */
    uint8_t  flags;          /* reserved */
    uint16_t tensor_id;      /* tensor identifier (0-65535) */
    uint32_t payload_len;    /* bytes following this header */
} edgecoh_header_t;

/* TRANSFER_OWNERSHIP: 8-byte header + 1 byte target device */
typedef struct __attribute__((packed)) {
    edgecoh_header_t header;
    uint8_t target_device;   /* edgecoh_device_t */
} edgecoh_transfer_msg_t;

/* PREFETCH: 8-byte header + 1 byte target device + 4 byte DDR2 offset */
typedef struct __attribute__((packed)) {
    edgecoh_header_t header;
    uint8_t  target_device;
    uint32_t ddr2_offset;    /* byte offset in FPGA DDR2 */
} edgecoh_prefetch_msg_t;

/* SYNC_BARRIER: header only (payload_len = 0) */
typedef edgecoh_header_t edgecoh_barrier_msg_t;

/* DATA_WRITE: header + DDR2 address + data bytes follow */
typedef struct __attribute__((packed)) {
    edgecoh_header_t header;
    uint32_t ddr2_addr;      /* destination address in FPGA DDR2 */
    /* payload_len bytes of tensor data follow */
} edgecoh_data_write_msg_t;

/* DATA_READ: header + DDR2 address + read length */
typedef struct __attribute__((packed)) {
    edgecoh_header_t header;
    uint32_t ddr2_addr;
    uint32_t read_len;       /* bytes to read */
} edgecoh_data_read_msg_t;

/* NMC_EXEC: header + operation type + operation-specific params */
typedef struct __attribute__((packed)) {
    edgecoh_header_t header;
    uint8_t  nmc_op;          /* edgecoh_nmc_op_t */
    uint32_t table_base_addr; /* DDR2 base address of weight/embedding table */
    uint32_t table_rows;      /* number of rows (embeddings) or input dim */
    uint32_t table_cols;      /* embedding dimension or output dim */
    uint32_t input_addr;      /* DDR2 address of input data (indices or activations) */
    uint32_t input_len;       /* number of input elements */
    uint32_t output_addr;     /* DDR2 address to write results */
} edgecoh_nmc_exec_msg_t;

/* Serialize a message to a byte buffer. Returns bytes written, or -1 on error. */
int edgecoh_serialize(const void *msg, uint8_t *buf, int buf_len);

/* Deserialize a message header from a byte buffer. Returns msg_type, or -1 on error. */
int edgecoh_deserialize_header(const uint8_t *buf, int buf_len, edgecoh_header_t *out);

#endif /* EDGECOH_MESSAGES_H */
