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
package com.integrallis.models.backend.purejava.edgerock;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.util.Objects;

/**
 * The Rock Memory Anchors half of an EdgeRock model: one memory module per full-attention layer.
 *
 * <p>EdgeRock's recurrent and attention halves are a Qwen3.5 hybrid in every shape that matters --
 * the published GGUF carries a complete {@code qwen35.*} key set beside its own, and every tensor
 * name in it is one the Qwen3.5 weight loader already resolves. This class holds the part that is
 * genuinely new, so the shared decoder needs one field rather than a parallel implementation.
 *
 * <p>Loaded eagerly. A memory module that failed to resolve mid-generation would leave the decoder
 * holding a half-written bank, and the residual contribution it feeds is not optional: without it
 * the model runs to completion and returns confident, wrong logits.
 */
public final class EdgeRockMemory {

  private final EdgeRockConfig config;

  /** Indexed by decoder layer; null on the recurrent layers, which carry no memory. */
  private final EdgeRockMemoryLayer[] layers;

  private EdgeRockMemory(EdgeRockConfig config, EdgeRockMemoryLayer[] layers) {
    this.config = config;
    this.layers = layers;
  }

  /** Resolves every memory module this file declares. */
  public static EdgeRockMemory load(GgufFile file) {
    return load(file, GgufBatchedMatrixKernel.none());
  }

  /**
   * Resolves every memory module, projecting through {@code kernel} where it is eligible.
   *
   * <p>The kernel matters more here than its nine projections per layer suggest. They are about a
   * tenth of the model's arithmetic per token but they ran on the calling thread, so before this
   * they set the decode rate: EdgeRock gained nothing from the native kernels while a control model
   * gained 3.9x.
   */
  public static EdgeRockMemory load(GgufFile file, GgufBatchedMatrixKernel kernel) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(kernel, "kernel");
    EdgeRockConfig config = EdgeRockConfig.fromMetadata(file.metadata());
    EdgeRockWeights weights = EdgeRockWeights.load(file, config, kernel);
    EdgeRockMemoryLayer[] layers = new EdgeRockMemoryLayer[config.blockCount()];
    for (int layer : config.fullAttentionLayers()) {
      layers[layer] = new EdgeRockMemoryLayer(config, weights.memoryLayer(layer));
    }
    return new EdgeRockMemory(config, layers);
  }

  public EdgeRockConfig config() {
    return config;
  }

  /** The memory module for this layer, or null when the layer carries none. */
  public EdgeRockMemoryLayer layer(int layer) {
    return layers[layer];
  }

  /**
   * Fresh per-layer banks for one session, null on the layers that carry no memory.
   *
   * <p>One bank per layer and per session: the bank is the model's state, so sharing an array
   * across sessions would let one sequence read another's anchors.
   */
  public MemoryBank[] newBanks() {
    MemoryBank[] banks = new MemoryBank[config.blockCount()];
    for (int layer : config.fullAttentionLayers()) {
      banks[layer] =
          new MemoryBank(
              config.headCountKv(),
              config.memoryCapacity(),
              config.memoryCommitChunk(),
              config.headDim());
    }
    return banks;
  }
}
