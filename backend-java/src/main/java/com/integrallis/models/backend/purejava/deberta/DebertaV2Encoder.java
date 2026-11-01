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
package com.integrallis.models.backend.purejava.deberta;

import com.integrallis.models.backend.purejava.tensor.TensorSource;
import java.util.Objects;

/** Shared DeBERTa encoder used by task heads without a sequence-classification pooler. */
public final class DebertaV2Encoder {
  private final DebertaV2ForwardPass forward;

  private DebertaV2Encoder(DebertaV2ForwardPass forward) {
    this.forward = forward;
  }

  /** Loads owned F32 encoder weights from the declared tensor namespace. */
  public static DebertaV2Encoder load(
      TensorSource source, DebertaV2Config config, String namespace) {
    Objects.requireNonNull(namespace, "namespace");
    return new DebertaV2Encoder(
        new DebertaV2ForwardPass(config, DebertaV2Weights.loadEncoder(source, config, namespace)));
  }

  /** Returns caller-owned final hidden states at the requested input positions. */
  public float[][] encode(int[] tokens, int[] positions) {
    return forward.encode(tokens, positions);
  }
}
