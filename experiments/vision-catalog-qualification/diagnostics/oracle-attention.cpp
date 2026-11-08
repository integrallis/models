// External diagnostic only: invoke the pinned independent ggml attention operator.
// This file and its linked library are never packaged in Models or used for admission timings.
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

extern "C" int oracle_attention(const float * query, const float * keys,
        const float * values, float * output, int batch, int start, int heads,
        int width, int capacity, int threads) {
    if (!query || !keys || !values || !output || batch < 1 || start < 0 ||
        heads < 1 || width < 1 || capacity < batch || start > capacity-batch ||
        threads < 1 || capacity > 4096 || heads > 128 || width > 512) return -1;
    // Match llama.cpp's Flash Attention KV padding and causal mask. Inputs are
    // batch-major Q and head-major F32 cache; its graph casts K/V to F16.
    const int positions = start + batch;
    const int padded = ((positions + 255) / 256) * 256;
    const size_t q_elements = size_t(batch)*heads*width;
    const size_t kv_elements = size_t(padded)*heads*width;
    const size_t memory = 32*1024*1024 + 16*q_elements + 8*kv_elements + size_t(padded)*batch*4;
    ggml_init_params params{memory, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    if (!ctx) return -2;
    auto q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, width, batch, heads, 1);
    auto k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, width, padded, heads, 1);
    auto v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, width, padded, heads, 1);
    // ggml requires the query axis of the mask padded to GGML_KQ_MASK_PAD.
    const int mask_rows = ((batch+63)/64)*64;
    auto mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, padded, mask_rows, 1, 1);
    auto qp = static_cast<float *>(q->data);
    auto kp = static_cast<ggml_fp16_t *>(k->data);
    auto vp = static_cast<ggml_fp16_t *>(v->data);
    std::fill_n(kp, kv_elements, ggml_fp32_to_fp16(0));
    std::fill_n(vp, kv_elements, ggml_fp32_to_fp16(0));
    for (int h=0; h<heads; ++h) {
        for (int row=0; row<batch; ++row)
            std::copy_n(query+(size_t(row)*heads+h)*width, width, qp+(size_t(h)*batch+row)*width);
        for (int row=0; row<positions; ++row) {
            auto source = (size_t(h)*capacity+row)*width;
            auto target = (size_t(h)*padded+row)*width;
            ggml_fp32_to_fp16_row(keys+source, kp+target, width);
            ggml_fp32_to_fp16_row(values+source, vp+target, width);
        }
    }
    auto mp = static_cast<ggml_fp16_t *>(mask->data);
    std::fill_n(mp, size_t(padded)*mask_rows, ggml_fp32_to_fp16(-INFINITY));
    for (int row=0; row<batch; ++row)
        std::fill_n(mp+size_t(row)*padded, start+row+1, ggml_fp32_to_fp16(0));
    auto result = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, float(1.0/std::sqrt(double(width))), 0, 0);
    auto graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, result);
    const int status = ggml_graph_compute_with_ctx(ctx.get(), graph, threads);
    if (status != GGML_STATUS_SUCCESS) return status == 0 ? -3 : status;
    std::memcpy(output, result->data, q_elements*sizeof(float));
    return 0;
}
