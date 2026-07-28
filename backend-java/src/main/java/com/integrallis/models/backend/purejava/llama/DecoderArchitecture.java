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

/** Decoder semantics implemented by the Java transformer graph. */
public enum DecoderArchitecture {
  LLAMA("llama"),
  QWEN2("qwen2"),
  QWEN3("qwen3"),
  SMOLLM3("smollm3"),
  GEMMA3("gemma3");

  private final String metadataId;

  DecoderArchitecture(String metadataId) {
    this.metadataId = metadataId;
  }

  /** Stable GGUF {@code general.architecture} value. */
  public String metadataId() {
    return metadataId;
  }

  /** Resolves only architectures whose complete decoder semantics are implemented. */
  public static DecoderArchitecture parse(String value) {
    String normalized = value == null ? LLAMA.metadataId : value.trim().toLowerCase(Locale.ROOT);
    for (DecoderArchitecture architecture : values()) {
      if (architecture.metadataId.equals(normalized)) {
        return architecture;
      }
    }
    throw new IllegalArgumentException(
        "Unsupported GGUF decoder architecture: "
            + normalized
            + "; supported architectures are llama, qwen2, qwen3, smollm3, gemma3");
  }
}
