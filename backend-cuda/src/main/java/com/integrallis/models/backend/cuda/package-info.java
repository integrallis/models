/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Models-owned GPU kernels for K-quant projections and decode attention.
 *
 * <p>Two kernels run on the device: a fused K-quant dequantise-and-multiply projection, and
 * grouped-query attention for the single-token decode step. Everything else — model parsing, the
 * tokenizer, the graph, norms, RoPE, residuals, KV cache ownership, activation quantisation,
 * sampling and the generation loop — stays in Java, exactly as {@code backend-native} states for
 * the CPU shim. There is no engine here, no vendor math library, and no inference runtime.
 *
 * <p>The kernels are written in Rust and compiled to PTX with rustc's in-tree {@code
 * nvptx64-nvidia-cuda} target; the host side is Panama FFM against the CUDA driver API. The build
 * packages the PTX inside the jar with a SHA-256 the loader recomputes, and every failure path — no
 * driver, old device, unsupported format, ineligible shape — falls back to the Vector API silently
 * and without cost.
 *
 * <p>See {@code backend-cuda/README.md} for what is verified off-device and what still needs a GPU
 * host, and {@code backend-cuda/UPSTREAM.md} for the toolchain limitations encountered, each with a
 * reproduction.
 */
package com.integrallis.models.backend.cuda;
