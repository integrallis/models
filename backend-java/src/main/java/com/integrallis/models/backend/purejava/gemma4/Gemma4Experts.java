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

import java.io.IOException;
import java.lang.foreign.MemorySegment;

/** Routed-expert weight access owned by one loaded Gemma 4 decoder. */
interface Gemma4Experts extends AutoCloseable {

  @FunctionalInterface
  interface ExpertResolver {
    Gemma4TensorLayout.ExpertWeights resolve(int layer, int expert);
  }

  interface Lease extends AutoCloseable {
    int layer();

    int expert();

    MemorySegment gateUp();

    MemorySegment down();

    @Override
    void close();
  }

  default Lease acquire(int layer, int expert) throws IOException {
    return acquire(layer, expert, 1);
  }

  Lease acquire(int layer, int expert, int useCount) throws IOException;

  int concurrentLeasesPerLayer();

  @Override
  void close() throws IOException;
}
