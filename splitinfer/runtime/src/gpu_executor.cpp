/* splitinfer/runtime/src/gpu_executor.cpp
 * TensorRT-based GPU executor implementing GpuExecutorBase.
 *
 * When built with SPLITINFER_HAS_TENSORRT, loads pre-built .engine files
 * and executes layers via TensorRT.
 * Without TensorRT the executor acts as a passthrough (logs a warning,
 * copies input to output where sizes match, returns true).
 */

#include "splitinfer/gpu_executor.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef SPLITINFER_HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime.h>

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

static TrtLogger& get_logger() {
    static TrtLogger logger;
    return logger;
}

struct EngineContext {
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    /* TensorRT 10.x removed the explicit destroy() methods that TRT 8.x used.
     * Objects are now destroyed via plain `delete` (standard C++ ownership). */
    ~EngineContext() {
        delete context;
        delete engine;
        delete runtime;
    }
};

} // anonymous namespace
#endif // SPLITINFER_HAS_TENSORRT

namespace splitinfer {

// ── Pimpl implementation ────────────────────────────────────────────────────

struct GpuExecutor::Impl {
#ifdef SPLITINFER_HAS_TENSORRT
    std::unordered_map<std::string, std::unique_ptr<EngineContext>> engines;
#endif
};

GpuExecutor::GpuExecutor() : impl_(std::make_unique<Impl>()) {}
GpuExecutor::~GpuExecutor() = default;

bool GpuExecutor::load_engine(const std::string& layer_name,
                               const std::string& engine_path) {
#ifdef SPLITINFER_HAS_TENSORRT
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "GpuExecutor: cannot open engine file '%s'\n",
                     engine_path.c_str());
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> blob(static_cast<size_t>(size));
    if (!file.read(blob.data(), size)) {
        std::fprintf(stderr, "GpuExecutor: failed to read engine file '%s'\n",
                     engine_path.c_str());
        return false;
    }

    auto ctx = std::make_unique<EngineContext>();
    ctx->runtime = nvinfer1::createInferRuntime(get_logger());
    if (!ctx->runtime) {
        std::fprintf(stderr, "GpuExecutor: createInferRuntime failed\n");
        return false;
    }

    ctx->engine = ctx->runtime->deserializeCudaEngine(blob.data(), blob.size());
    if (!ctx->engine) {
        std::fprintf(stderr, "GpuExecutor: deserializeCudaEngine failed for '%s'\n",
                     engine_path.c_str());
        return false;
    }

    ctx->context = ctx->engine->createExecutionContext();
    if (!ctx->context) {
        std::fprintf(stderr, "GpuExecutor: createExecutionContext failed for '%s'\n",
                     engine_path.c_str());
        return false;
    }

    impl_->engines[layer_name] = std::move(ctx);
    std::fprintf(stdout, "GpuExecutor: loaded engine for layer '%s' from '%s'\n",
                 layer_name.c_str(), engine_path.c_str());
    return true;
#else
    (void)layer_name;
    (void)engine_path;
    std::fprintf(stderr, "GpuExecutor: TensorRT not available — cannot load engine\n");
    return false;
#endif
}

bool GpuExecutor::execute(const std::string& layer_name,
                           const void* input, size_t input_bytes,
                           void* output, size_t output_bytes) {
#ifdef SPLITINFER_HAS_TENSORRT
    auto it = impl_->engines.find(layer_name);
    if (it == impl_->engines.end()) {
        std::fprintf(stderr, "GpuExecutor: no engine loaded for layer '%s' — passthrough\n",
                     layer_name.c_str());
        if (output && input && output_bytes > 0) {
            size_t copy_len = (input_bytes < output_bytes) ? input_bytes : output_bytes;
            std::memcpy(output, input, copy_len);
        }
        return true;
    }

    auto& ctx = it->second;

    void* d_input  = nullptr;
    void* d_output = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_input, input_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "GpuExecutor: cudaMalloc input failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc(&d_output, output_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "GpuExecutor: cudaMalloc output failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_input);
        return false;
    }

    err = cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "GpuExecutor: cudaMemcpy H2D failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    // TensorRT 10.x: use the named-tensor API instead of the index-based
    // binding API.  Iterate IO tensors by name, find the input and output
    // by their TensorIOMode, and bind device pointers.  Then call enqueueV3
    // (executeV2 still works but is deprecated in TRT 10).
    int nb = ctx->engine->getNbIOTensors();
    bool bound_input = false, bound_output = false;
    for (int i = 0; i < nb; ++i) {
        const char* name = ctx->engine->getIOTensorName(i);
        auto mode = ctx->engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            ctx->context->setTensorAddress(name, d_input);
            bound_input = true;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            ctx->context->setTensorAddress(name, d_output);
            bound_output = true;
        }
    }
    if (!bound_input || !bound_output) {
        std::fprintf(stderr,
            "GpuExecutor: engine for layer '%s' missing input or output tensor (in=%d, out=%d)\n",
            layer_name.c_str(), bound_input, bound_output);
        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    // enqueueV3 needs a CUDA stream; we use the default stream (0) for synchronous behavior.
    bool ok = ctx->context->enqueueV3(0);
    if (ok) {
        // enqueueV3 is async on the stream — synchronize before reading output.
        cudaStreamSynchronize(0);
    }
    if (!ok) {
        std::fprintf(stderr, "GpuExecutor: TensorRT executeV2 failed for layer '%s'\n",
                     layer_name.c_str());
        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    err = cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    if (err != cudaSuccess) {
        std::fprintf(stderr, "GpuExecutor: cudaMemcpy D2H failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    return true;
#else
    // No TensorRT: passthrough — copy input to output and warn once.
    static bool warned = false;
    if (!warned) {
        std::fprintf(stderr,
            "GpuExecutor: TensorRT not available — running as passthrough for all GPU layers\n");
        warned = true;
    }
    std::fprintf(stdout, "  [GPU passthrough] %s  in=%zu B  out=%zu B\n",
                 layer_name.c_str(), input_bytes, output_bytes);
    if (output && input && output_bytes > 0) {
        size_t copy_len = (input_bytes < output_bytes) ? input_bytes : output_bytes;
        std::memcpy(output, input, copy_len);
    }
    return true;
#endif
}

} // namespace splitinfer
