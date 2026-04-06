#ifndef SPLITINFER_FPGA_EXECUTOR_H
#define SPLITINFER_FPGA_EXECUTOR_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace splitinfer {

/// Abstract base class for FPGA layer execution via EdgeCoh/NMC.
class FpgaExecutorBase {
public:
    virtual ~FpgaExecutorBase() = default;

    /// Execute an NMC operation on the FPGA.
    /// @param layer_name Logical name of the layer.
    /// @param nmc_op     NMC operation code (device-specific).
    /// @param input      Pointer to input data in host memory.
    /// @param input_bytes Size of input in bytes.
    /// @param output     Pointer to pre-allocated output buffer (host memory).
    /// @param output_bytes Size of output buffer in bytes.
    /// @return true on success.
    virtual bool execute(const std::string& layer_name,
                         uint32_t nmc_op,
                         const void* input, size_t input_bytes,
                         void* output, size_t output_bytes) = 0;

    /// Copy a region of FPGA memory to host memory.
    /// @param fpga_addr Source address in FPGA memory space.
    /// @param len       Number of bytes to copy.
    /// @param dst       Destination host buffer.
    /// @return true on success.
    virtual bool transfer_to_host(uint32_t fpga_addr, size_t len, void* dst) = 0;

    /// Copy host memory to a region of FPGA memory.
    /// @param src       Source host buffer.
    /// @param len       Number of bytes to copy.
    /// @param fpga_addr Destination address in FPGA memory space.
    /// @return true on success.
    virtual bool transfer_to_fpga(const void* src, size_t len, uint32_t fpga_addr) = 0;
};

} // namespace splitinfer
#endif
