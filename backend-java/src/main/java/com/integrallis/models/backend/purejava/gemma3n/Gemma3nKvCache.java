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
package com.integrallis.models.backend.purejava.gemma3n;

import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.LayerSpec;
import java.util.Objects;

/**
 * Key-value cache layout for Gemma 3n.
 *
 * <p>A sliding layer gets a ring buffer sized to its window plus the prefill batch, and a
 * full-attention layer a linear one. The layers that <b>share</b> another layer's cache are
 * allocated nothing: they read the owner's buffer, and giving them one of their own would quietly
 * let a bug write there and still produce output.
 */
final class Gemma3nKvCache {

  private Gemma3nKvCache() {}

  static LayeredKvCache create(
      Gemma3nConfig config, int runtimeContextLength, int prefillCapacity) {
    Objects.requireNonNull(config, "config");
    positive("runtimeContextLength", runtimeContextLength);
    positive("prefillCapacity", prefillCapacity);
    if (runtimeContextLength > config.contextLength()) {
      throw new IllegalArgumentException(
          "runtimeContextLength "
              + runtimeContextLength
              + " exceeds model context length "
              + config.contextLength());
    }

    int ringCapacity = Math.addExact(config.slidingWindow(), prefillCapacity);
    LayerSpec[] layers = new LayerSpec[config.kvOwningLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      layers[layer] =
          config.usesSlidingWindow(layer)
              ? LayerSpec.ring(config.keyDim(), config.keyDim(), ringCapacity)
              : LayerSpec.linear(config.keyDim(), config.keyDim());
    }
    return new LayeredKvCache(runtimeContextLength, layers);
  }

  private static void positive(String name, int value) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be > 0: " + value);
    }
  }
}
