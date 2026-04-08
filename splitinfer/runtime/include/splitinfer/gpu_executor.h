#ifndef SPLITINFER_GPU_EXECUTOR_H
#define SPLITINFER_GPU_EXECUTOR_H

#include <cstddef>
#include <memory>
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

/// Concrete GPU executor. Without SPLITINFER_HAS_TENSORRT, acts as a passthrough.
class GpuExecutor : public GpuExecutorBase {
public:
    GpuExecutor();
    ~GpuExecutor() override;

    /// Load a serialized TensorRT engine file for a named layer.
    bool load_engine(const std::string& layer_name, const std::string& engine_path);

    bool execute(const std::string& layer_name,
                 const void* input, size_t input_bytes,
                 void* output, size_t output_bytes) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace splitinfer
#endif
