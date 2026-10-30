// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0
//
// Independent numerical oracle using llama.cpp's public C API.
// Usage: reference-logits MODEL.gguf 3,4,5 [GREEDY_TOKENS]
//        reference-logits MODEL.gguf 'text:<|user|>Hello<|assistant|>' [GREEDY_TOKENS]
// Build and reference revision are documented in docs/flint.md.
#include "llama.h"
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false; // Keep expert weights mapped; avoid a second repacked copy.
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.n_threads = 6;
    cp.n_threads_batch = 6;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    const auto * vocab = llama_model_get_vocab(model);
    const int width = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> tokens;
    std::string input = argv[2];
    if (input.rfind("text:", 0) == 0) {
        input.erase(0, 5);
        int count = -llama_tokenize(vocab, input.data(), input.size(), nullptr, 0, false, true);
        tokens.resize(count);
        count = llama_tokenize(vocab, input.data(), input.size(), tokens.data(), tokens.size(), false, true);
        if (count < 0) return 5;
        tokens.resize(count);
    } else {
        std::stringstream stream(input);
        std::string part;
        while (std::getline(stream, part, ',')) tokens.push_back(std::stoi(part));
    }
    int prompt = tokens.size();
    int generated = argc > 3 ? std::stoi(argv[3]) : 0;
    std::cout << std::setprecision(9) << "{\"prompt_tokens\":[";
    for (int i = 0; i < prompt; ++i) std::cout << (i ? "," : "") << tokens[i];
    std::cout << "],\"rows\":[";
    for (int pos = 0; pos < prompt + generated; ++pos) {
        auto batch = llama_batch_get_one(&tokens[pos], 1);
        if (llama_decode(ctx, batch)) return 6;
        const float * logits = llama_get_logits_ith(ctx, -1);
        int top = std::max_element(logits, logits + width) - logits;
        std::cout << (pos ? "," : "") << "{\"position\":" << pos << ",\"input\":"
                  << tokens[pos] << ",\"top\":" << top << ",\"logits\":[";
        // Full vectors for the tiny fixture; top logits for the real checkpoint.
        std::vector<int> ids(width);
        for (int i = 0; i < width; ++i) ids[i] = i;
        if (width > 256) {
            std::partial_sort(ids.begin(), ids.begin() + 16, ids.end(),
                [logits](int a, int b) { return logits[a] > logits[b]; });
            ids.resize(16);
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << (i ? "," : "") << "[" << ids[i] << "," << logits[ids[i]] << "]";
        }
        std::cout << "]}" << std::flush;
        if (pos >= prompt - 1) tokens.push_back(top);
    }
    std::cout << "]}\n";
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
