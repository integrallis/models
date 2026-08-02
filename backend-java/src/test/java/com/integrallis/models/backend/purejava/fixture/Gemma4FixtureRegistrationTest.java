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
package com.integrallis.models.backend.purejava.fixture;

import static org.assertj.core.api.Assertions.assertThat;

import java.net.URI;
import org.junit.jupiter.api.Test;

class Gemma4FixtureRegistrationTest {

  private static final ModelFixtureRequirement GEMMA4_26B_A4B_Q4_K_M =
      ModelFixtureRequirement.of("hf://ggml-org/gemma-4-26B-A4B-it-GGUF")
          .version("[4.0.0,5.0.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("chat");

  @Test
  void resolvesTheExactModelJarsArtifact() {
    ModelFixtureDescriptor descriptor =
        ModelFixtureRegistry.fromClasspath().resolve(GEMMA4_26B_A4B_Q4_K_M).orElseThrow();

    assertThat(descriptor.id()).isEqualTo("gemma4_26b_a4b_it_q4_k_m");
    assertThat(descriptor.downloadUri())
        .isEqualTo(
            URI.create(
                "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/"
                    + "ae4d537a6345467d1c86bb5cc0d4505ff3ebe0f3/"
                    + "gemma-4-26B-A4B-it-Q4_K_M.gguf"));
    assertThat(descriptor.sha256())
        .contains("88f4a13b0bb95f031a7fad973e10854122fb67ebc34d214d39a2f65053046abc");
    assertThat(descriptor.sizeBytes()).contains(16_796_015_136L);
    assertThat(descriptor.architecture()).isEqualTo("gemma4");
    assertThat(descriptor.quantization()).isEqualTo("Q4_K_M");
    assertThat(descriptor.features())
        .contains("mixture-of-experts", "expert-streaming", "hybrid-attention", "chat-template");
    assertThat(descriptor.slow()).isTrue();
  }
}
