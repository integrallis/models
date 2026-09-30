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

/**
 * Hands the toy Gemma 3n graph to tests in the backend package.
 *
 * <p>The weights loader and the cache layout are package-private on purpose -- nothing outside this
 * package should build them -- but the decoder adapter lives one package up and needs a graph to
 * adapt. One narrow door rather than widening either of those.
 */
public final class Gemma3nTestAccess {

  private Gemma3nTestAccess() {}

  /** The toy model as a ready graph, over a cache with room for sixteen positions. */
  public static Gemma3nForwardPass toyForwardPass() {
    Gemma3nToyModel model = Gemma3nToyModel.create();
    Gemma3nConfig config = Gemma3nToyModel.config();
    return new Gemma3nForwardPass(
        config,
        Gemma3nWeights.fromGgufFile(model.file(), config),
        Gemma3nKvCache.create(config, 16, 1));
  }
}
