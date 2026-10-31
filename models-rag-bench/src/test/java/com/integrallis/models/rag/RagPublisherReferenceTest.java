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
package com.integrallis.models.rag;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** The reviewed publisher references, and the exact strings a comparator report must carry. */
@Tag("unit")
class RagPublisherReferenceTest {

  @Test
  void bonsaiBackendVersionIsUnchangedByGeneralisingTheSlug() {
    // The hard-coded "PrismML-Eng/llama.cpp@" prefix became a per-entry field so EdgeRock could
    // pin a different upstream. This asserts the refactor did not move Bonsai's string by one
    // character: a comparator report stamped with the old value must still match, and the catalog
    // cannot shrink.
    assertThat(RagPublisherReference.BONSAI2_PTQ1_0.backendVersion())
        .isEqualTo("PrismML-Eng/llama.cpp@e8fd209c975c80aa7fb36a8423571ff3b1b7e955");
    assertThat(RagPublisherReference.BONSAI2_PTQ1_0.patchId()).isEmpty();
  }

  @Test
  void edgeRockPinsThePatchBecauseTheRevisionAloneDoesNotServeTheArchitecture() {
    RagPublisherReference reference = RagPublisherReference.EDGEROCK_4B_Q4_K_M;

    // Upstream llama.cpp at this commit has no edgerock architecture; the patch is what adds it,
    // so a reference naming only the commit would describe a build that cannot load the model.
    assertThat(reference.backendVersion())
        .isEqualTo(
            "ggml-org/llama.cpp@62318cd57dc63900ecb153272c0b10f78550855d"
                + "+edgerock-rma@6a4c32359d88c84a8cee2a6920847220da61d7931d406dc2c0970f0da9dbf20f");
    assertThat(reference.artifactSha256())
        .isEqualTo("67c354bcb2abe0cba407fbfe8e7f7cc100dd44458b6fbdfee95ae22f3aaa2c6a");
    assertThat(reference.artifactSizeBytes()).isEqualTo(2_936_478_592L);
    assertThat(reference.modelId()).isEqualTo("hkust_aaron_edgerock_4b_gguf_q4_k_m");
  }

  @Test
  void eachReferenceIsReachableByItsOwnId() {
    for (RagPublisherReference reference : RagPublisherReference.values()) {
      assertThat(RagPublisherReference.fromId(reference.id())).isSameAs(reference);
    }
    assertThatThrownBy(() -> RagPublisherReference.fromId("not-reviewed"))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
