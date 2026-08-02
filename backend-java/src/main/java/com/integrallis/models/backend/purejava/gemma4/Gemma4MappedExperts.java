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

import com.integrallis.models.backend.purejava.gemma4.Gemma4TensorLayout.ExpertWeights;
import com.integrallis.models.backend.purejava.gemma4.Gemma4TensorLayout.TensorSlice;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/** Zero-copy routed experts backed by slices of the decoder's mapped GGUF. */
final class Gemma4MappedExperts implements Gemma4Experts {

  private final MappedLease[][] experts;
  private final AtomicBoolean closed = new AtomicBoolean();

  Gemma4MappedExperts(
      MemorySegment mappedFile, int numLayers, int numExperts, ExpertResolver resolver) {
    Objects.requireNonNull(mappedFile, "mappedFile");
    Objects.requireNonNull(resolver, "resolver");
    if (numLayers <= 0 || numExperts <= 0) {
      throw new IllegalArgumentException("numLayers and numExperts must be positive");
    }
    this.experts = new MappedLease[numLayers][numExperts];
    for (int layer = 0; layer < numLayers; layer++) {
      for (int expert = 0; expert < numExperts; expert++) {
        ExpertWeights weights =
            Objects.requireNonNull(resolver.resolve(layer, expert), "resolved expert weights");
        if (weights.layer() != layer || weights.expert() != expert) {
          throw new IllegalArgumentException("expert resolver returned mismatched coordinates");
        }
        experts[layer][expert] =
            new MappedLease(
                weights, slice(mappedFile, weights.gateUp()), slice(mappedFile, weights.down()));
      }
    }
  }

  @Override
  public Lease acquire(int layer, int expert, int useCount) {
    if (closed.get()) {
      throw new IllegalStateException("Gemma 4 mapped experts are closed");
    }
    if (layer < 0 || layer >= experts.length) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
    if (expert < 0 || expert >= experts[layer].length) {
      throw new IllegalArgumentException("expert out of range: " + expert);
    }
    if (useCount <= 0) {
      throw new IllegalArgumentException("useCount must be > 0: " + useCount);
    }
    return experts[layer][expert];
  }

  @Override
  public int concurrentLeasesPerLayer() {
    return experts[0].length;
  }

  @Override
  public void close() {
    closed.set(true);
  }

  private static MemorySegment slice(MemorySegment mappedFile, TensorSlice source) {
    return mappedFile.asSlice(source.fileOffset(), source.byteSize());
  }

  private record MappedLease(ExpertWeights weights, MemorySegment gateUp, MemorySegment down)
      implements Lease {

    @Override
    public int layer() {
      return weights.layer();
    }

    @Override
    public int expert() {
      return weights.expert();
    }

    @Override
    public void close() {}
  }
}
