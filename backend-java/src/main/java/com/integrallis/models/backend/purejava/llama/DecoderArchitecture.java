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
package com.integrallis.models.backend.purejava.llama;

import java.util.Locale;

/**
 * Transformer semantics implemented by the Java graph.
 *
 * <p>Listing an architecture here asserts that its semantics are implemented end to end. An entry
 * added ahead of its implementation is worse than no entry at all: the backend would accept the
 * model and emit activations computed by the wrong graph, with nothing anywhere to fail on.
 *
 * <p>Most entries are causal decoders. {@link #GEMMA_EMBEDDING} is not — it is an encoder and runs
 * a separate pass; {@link LlamaConfig#usesBidirectionalAttention()} is what tells them apart.
 */
public enum DecoderArchitecture {
  LLAMA("llama"),
  QWEN2("qwen2"),
  QWEN3("qwen3"),
  SMOLLM3("smollm3"),
  GEMMA3("gemma3"),
  GEMMA_EMBEDDING("gemma-embedding");

  private final String metadataId;

  DecoderArchitecture(String metadataId) {
    this.metadataId = metadataId;
  }

  /** Stable GGUF {@code general.architecture} value. */
  public String metadataId() {
    return metadataId;
  }

  /** Resolves only architectures whose complete semantics are implemented. */
  public static DecoderArchitecture parse(String value) {
    String normalized = value == null ? LLAMA.metadataId : value.trim().toLowerCase(Locale.ROOT);
    for (DecoderArchitecture architecture : values()) {
      if (architecture.metadataId.equals(normalized)) {
        return architecture;
      }
    }
    StringBuilder supported = new StringBuilder();
    for (DecoderArchitecture architecture : values()) {
      supported.append(supported.isEmpty() ? "" : ", ").append(architecture.metadataId);
    }
    throw new IllegalArgumentException(
        "Unsupported GGUF architecture: "
            + normalized
            + "; supported architectures are "
            + supported);
  }
}
