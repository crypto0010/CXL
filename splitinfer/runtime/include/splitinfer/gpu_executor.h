#ifndef SPLITINFER_GPU_EXECUTOR_H
#define SPLITINFER_GPU_EXECUTOR_H

#include <cstddef>
#include <string>

namespace splitinfer {

/// Abstract base class for GPU layer execution.
class GpuExecutorBase {
public:
    virtual ~GpuExecutorBase() = default;

    /// Execute a single layer on the GPU.
    /// @param layer_name  Logical name of the layer.
    /// @param input       Pointer to input tensor data (host or device memory).
    /// @param input_bytes Size of input buffer in bytes.
    /// @param output      Pointer to pre-allocated output buffer.
    /// @param output_bytes Size of output buffer in bytes.
    /// @return true on success.
    virtual bool execute(const std::string& layer_name,
                         const void* input, size_t input_bytes,
                         void* output, size_t output_bytes) = 0;
};

} // namespace splitinfer
#endif
