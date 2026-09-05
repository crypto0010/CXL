#include "executor.h"
#include <algorithm>
#include <cstring>

namespace splitinfer2 { namespace kernels {

void gather(const int8_t* table, int dim_bytes, int dim, int32_t idx, int8_t* out) {
    std::memcpy(out, table + (size_t)idx * dim_bytes, (size_t)dim);
}

void fc_acc(const int8_t* W, int M_pad, int K_pad, const int8_t* x, int32_t* acc) {
    for (int m = 0; m < M_pad; m++) {
        const int8_t* row = W + (size_t)m * K_pad;
        int32_t a = 0;
        for (int k = 0; k < K_pad; k++) a += (int32_t)row[k] * (int32_t)x[k];
        acc[m] = a;
    }
}

void epilogue(const int32_t* acc, const int32_t* bias, int M_pad, int mult, int shift, int relu, int8_t* out) {
    for (int i = 0; i < M_pad; i++) {
        int64_t s = (int64_t)acc[i] + (int64_t)bias[i];
        int64_t p = s * (int64_t)mult;
        int64_t sh = p >> shift;               /* arithmetic, matches RTL >>> */
        if (relu && sh < 0) sh = 0;
        out[i] = (int8_t)std::max<int64_t>(-128, std::min<int64_t>(127, sh));
    }
}
}}
