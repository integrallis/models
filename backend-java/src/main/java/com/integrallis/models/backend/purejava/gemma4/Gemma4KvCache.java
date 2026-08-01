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
package com.integrallis.models.backend.purejava.gemma4;

import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.LayerSpec;
import java.util.Objects;

/** Constructs bounded per-layer KV state from Gemma 4 attention metadata. */
final class Gemma4KvCache {

  private Gemma4KvCache() {}

  static LayeredKvCache create(Gemma4Config config, int runtimeContextLength, int prefillCapacity) {
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
    LayerSpec[] layers = new LayerSpec[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      int keyDim = config.keyDim(layer);
      int valueDim = config.valueDim(layer);
      layers[layer] =
          config.usesSlidingWindow(layer)
              ? LayerSpec.ring(keyDim, valueDim, ringCapacity)
              : LayerSpec.linear(keyDim, valueDim);
    }
    return new LayeredKvCache(runtimeContextLength, layers);
  }

  private static void positive(String name, int value) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be > 0: " + value);
    }
  }
}
