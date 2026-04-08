/* splitinfer/runtime/src/fpga_executor.cpp
 * EdgeCoh-based FPGA executor implementing FpgaExecutorBase.
 *
 * Communicates with the Nexys 4 DDR FPGA over USB via the EdgeCoh
 * C library (libedgecoh).  Sends NMC_EXEC commands and data transfers
 * using the EdgeCoh message protocol.
 */

#include "splitinfer/fpga_executor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

extern "C" {
#include <edgecoh/transport.h>
#include <edgecoh/messages.h>
#include <edgecoh/edgecoh.h>
}

namespace splitinfer {

namespace {

/// FTDI FT2232HQ on Nexys 4 DDR board.
constexpr uint16_t DEFAULT_VID = 0x0403;
constexpr uint16_t DEFAULT_PID = 0x6010;

/// Maximum payload per USB transfer chunk (bytes).
constexpr size_t USB_CHUNK_SIZE = 4096;

/// Default timeout for receiving a response from the FPGA (ms).
constexpr int RECV_TIMEOUT_MS = 5000;

/// Drain leftover bytes from the USB receive buffer.
/// Used when a response payload exceeds the caller's output buffer.
static void drain_excess(edgecoh_transport_t* t, size_t bytes) {
    uint8_t tmp[256];
    size_t drained = 0;
    while (drained < bytes) {
        int want = static_cast<int>(std::min(bytes - drained, sizeof(tmp)));
        int got = edgecoh_transport_recv(t, tmp, want, RECV_TIMEOUT_MS);
        if (got <= 0) break;
        drained += static_cast<size_t>(got);
    }
}

/// Send exactly `len` bytes, checking for short writes.
static bool send_exact(edgecoh_transport_t* t, const uint8_t* buf, int len) {
    int sent = edgecoh_transport_send(t, buf, len);
    if (sent != len) {
        std::fprintf(stderr, "FpgaExecutor: short write — sent %d of %d bytes\n",
                     sent, len);
        return false;
    }
    return true;
}

} // anonymous namespace

// ── Pimpl implementation ────────────────────────────────────────────────────

struct FpgaExecutor::Impl {
    edgecoh_transport_t* transport = nullptr;
    uint16_t vid = DEFAULT_VID;
    uint16_t pid = DEFAULT_PID;
};

FpgaExecutor::FpgaExecutor() : impl_(std::make_unique<Impl>()) {}

FpgaExecutor::~FpgaExecutor() { close(); }

bool FpgaExecutor::open(uint16_t vid, uint16_t pid) {
    if (impl_->transport) return true;  // already open
    impl_->vid = vid;
    impl_->pid = pid;
    impl_->transport = edgecoh_transport_open(vid, pid);
    if (!impl_->transport) {
        std::fprintf(stderr, "FpgaExecutor: failed to open USB transport "
                     "(VID=0x%04X PID=0x%04X)\n", vid, pid);
        return false;
    }
    std::fprintf(stdout, "FpgaExecutor: USB transport opened\n");
    return true;
}

void FpgaExecutor::close() {
    if (impl_->transport) {
        edgecoh_transport_close(impl_->transport);
        impl_->transport = nullptr;
        std::fprintf(stdout, "FpgaExecutor: USB transport closed\n");
    }
}

bool FpgaExecutor::is_open() const {
    return impl_->transport != nullptr;
}

bool FpgaExecutor::execute(const std::string& layer_name,
                            uint32_t nmc_op,
                            const void* input, size_t input_bytes,
                            void* output, size_t output_bytes) {
    if (!impl_->transport && !open()) return false;

    // Build NMC_EXEC message.
    edgecoh_nmc_exec_msg_t msg;
    std::memset(&msg, 0, sizeof(msg));
    msg.header.msg_type    = EDGECOH_MSG_NMC_EXEC;
    msg.header.flags       = 0;
    msg.header.tensor_id   = 0;
    msg.header.payload_len = sizeof(msg) - sizeof(edgecoh_header_t);
    msg.nmc_op             = static_cast<uint8_t>(nmc_op);
    // Note: table_base_addr, table_rows, table_cols, output_addr are
    // set to 0 here.  The caller must pre-load weights via transfer_to_fpga()
    // and the FPGA firmware uses fixed address mappings per NMC op type.
    // A future enhancement can add a configure_layer() method to set
    // per-layer DDR2 address mappings.
    msg.input_addr         = 0;
    msg.input_len          = static_cast<uint32_t>(input_bytes);
    msg.output_addr        = 0;

    // Serialize and send.
    uint8_t buf[128];
    int ser_len = edgecoh_serialize(&msg, buf, sizeof(buf));
    if (ser_len < 0) {
        std::fprintf(stderr, "FpgaExecutor: serialize NMC_EXEC failed for layer '%s'\n",
                     layer_name.c_str());
        return false;
    }

    if (!send_exact(impl_->transport, buf, ser_len)) {
        std::fprintf(stderr, "FpgaExecutor: send NMC_EXEC failed for layer '%s'\n",
                     layer_name.c_str());
        return false;
    }

    // Wait for completion response from FPGA.
    // NOTE: The FPGA's edgecoh_controller.v uniformly sends MSG_ACK (0xFE)
    // upon completion of any dispatched message (SYNC_BARRIER, NMC_EXEC, etc.),
    // not a distinct NMC_DONE message.  We match the firmware's actual behavior.
    edgecoh_header_t hdr;
    int rc = edgecoh_recv_header(impl_->transport, &hdr, RECV_TIMEOUT_MS);
    if (rc < 0) {
        std::fprintf(stderr, "FpgaExecutor: recv ACK timed out for layer '%s'\n",
                     layer_name.c_str());
        return false;
    }

    if (hdr.msg_type != EDGECOH_MSG_ACK && hdr.msg_type != EDGECOH_MSG_NMC_DONE) {
        std::fprintf(stderr,
            "FpgaExecutor: expected ACK (0x%02X) or NMC_DONE (0x%02X), got 0x%02X for layer '%s'\n",
            EDGECOH_MSG_ACK, EDGECOH_MSG_NMC_DONE, hdr.msg_type, layer_name.c_str());
        return false;
    }

    // Read inline output payload if present.
    if (hdr.payload_len > 0) {
        size_t to_read = std::min(static_cast<size_t>(hdr.payload_len), output_bytes);
        if (output && to_read > 0) {
            int recvd = edgecoh_transport_recv(impl_->transport,
                                               static_cast<uint8_t*>(output),
                                               static_cast<int>(to_read),
                                               RECV_TIMEOUT_MS);
            if (recvd < 0) {
                std::fprintf(stderr,
                    "FpgaExecutor: recv NMC_DONE payload failed for layer '%s'\n",
                    layer_name.c_str());
                return false;
            }
        }
        // Drain any excess bytes to keep protocol stream aligned.
        size_t consumed = (output && output_bytes > 0) ? std::min(static_cast<size_t>(hdr.payload_len), output_bytes) : 0;
        if (hdr.payload_len > consumed) {
            drain_excess(impl_->transport, hdr.payload_len - consumed);
        }
    }

    return true;
}

bool FpgaExecutor::transfer_to_fpga(const void* src, size_t len, uint32_t fpga_addr) {
    if (!impl_->transport && !open()) return false;
    if (!src || len == 0) return true;

    const uint8_t* data = static_cast<const uint8_t*>(src);
    size_t remaining = len;
    uint32_t addr = fpga_addr;

    while (remaining > 0) {
        size_t chunk = std::min(remaining, USB_CHUNK_SIZE);

        // Build DATA_WRITE message.
        edgecoh_data_write_msg_t msg;
        std::memset(&msg, 0, sizeof(msg));
        msg.header.msg_type    = EDGECOH_MSG_DATA_WRITE;
        msg.header.flags       = 0;
        msg.header.tensor_id   = 0;
        msg.header.payload_len = static_cast<uint32_t>(sizeof(uint32_t) + chunk);  // ddr2_addr + tensor data
        msg.ddr2_addr          = addr;

        // Build a single contiguous buffer: serialized struct + payload data.
        // This prevents UART stream desync from two separate USB sends.
        uint8_t hdr_buf[64];
        int ser_len = edgecoh_serialize(&msg, hdr_buf, sizeof(hdr_buf));
        if (ser_len < 0) {
            std::fprintf(stderr, "FpgaExecutor: serialize DATA_WRITE failed\n");
            return false;
        }

        std::vector<uint8_t> combined(static_cast<size_t>(ser_len) + chunk);
        std::memcpy(combined.data(), hdr_buf, static_cast<size_t>(ser_len));
        std::memcpy(combined.data() + ser_len, data, chunk);

        if (!send_exact(impl_->transport, combined.data(), static_cast<int>(combined.size()))) {
            std::fprintf(stderr, "FpgaExecutor: send DATA_WRITE failed\n");
            return false;
        }

        // Wait for ACK.
        edgecoh_header_t hdr;
        int rc = edgecoh_recv_header(impl_->transport, &hdr, RECV_TIMEOUT_MS);
        if (rc < 0 || hdr.msg_type == EDGECOH_MSG_ERROR) {
            std::fprintf(stderr,
                "FpgaExecutor: DATA_WRITE ack failed (addr=0x%08X, chunk=%zu)\n",
                addr, chunk);
            return false;
        }

        data      += chunk;
        addr      += static_cast<uint32_t>(chunk);
        remaining -= chunk;
    }

    return true;
}

bool FpgaExecutor::transfer_to_host(uint32_t fpga_addr, size_t len, void* dst) {
    if (!impl_->transport && !open()) return false;
    if (!dst || len == 0) return true;

    // Build DATA_READ message.
    edgecoh_data_read_msg_t msg;
    std::memset(&msg, 0, sizeof(msg));
    msg.header.msg_type    = EDGECOH_MSG_DATA_READ;
    msg.header.flags       = 0;
    msg.header.tensor_id   = 0;
    msg.header.payload_len = sizeof(uint32_t) + sizeof(uint32_t); // ddr2_addr + read_len
    msg.ddr2_addr          = fpga_addr;
    msg.read_len           = static_cast<uint32_t>(len);

    uint8_t buf[64];
    int ser_len = edgecoh_serialize(&msg, buf, sizeof(buf));
    if (ser_len < 0) {
        std::fprintf(stderr, "FpgaExecutor: serialize DATA_READ failed\n");
        return false;
    }

    if (!send_exact(impl_->transport, buf, ser_len)) {
        std::fprintf(stderr, "FpgaExecutor: send DATA_READ failed\n");
        return false;
    }

    // Wait for DATA_RESPONSE header.
    edgecoh_header_t hdr;
    int rc = edgecoh_recv_header(impl_->transport, &hdr, RECV_TIMEOUT_MS);
    if (rc < 0) {
        std::fprintf(stderr, "FpgaExecutor: recv DATA_RESPONSE header timed out\n");
        return false;
    }

    if (hdr.msg_type != EDGECOH_MSG_DATA_RESPONSE) {
        std::fprintf(stderr,
            "FpgaExecutor: expected DATA_RESPONSE (0x%02X) but got 0x%02X\n",
            EDGECOH_MSG_DATA_RESPONSE, hdr.msg_type);
        return false;
    }

    // Receive the payload data in chunks.
    size_t to_read = std::min(static_cast<size_t>(hdr.payload_len), len);
    size_t total_read = 0;
    uint8_t* out = static_cast<uint8_t*>(dst);

    while (total_read < to_read) {
        int want = static_cast<int>(std::min(to_read - total_read, USB_CHUNK_SIZE));
        int recvd = edgecoh_transport_recv(impl_->transport,
                                           out + total_read,
                                           want,
                                           RECV_TIMEOUT_MS);
        if (recvd <= 0) {
            std::fprintf(stderr,
                "FpgaExecutor: recv DATA_RESPONSE payload failed (%zu/%zu bytes)\n",
                total_read, to_read);
            return false;
        }
        total_read += static_cast<size_t>(recvd);
    }

    // Drain excess if FPGA sent more than we needed.
    if (hdr.payload_len > to_read) {
        drain_excess(impl_->transport, hdr.payload_len - to_read);
    }

    return true;
}

} // namespace splitinfer
