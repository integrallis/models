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
package com.integrallis.models.backend.apple;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class AppleFoundationModelsStubTest {

  @Test
  void stubClientReportsAvailableAndGeneratesDeterministicText() {
    try (AppleFoundationModelsClient client = AppleFoundationModels.createStub()) {
      AppleFoundationModelsAvailability availability = client.availability();

      assertThat(availability.supported()).isTrue();
      assertThat(availability.available()).isTrue();
      assertThat(availability.reason()).contains("stub");
      assertThat(client.generate("Reply with the single word hello.").text()).isEqualTo("hello");
    }
  }

  @Test
  void createCanBeForcedToUseStubModeBeforePlatformGate() {
    String previous = System.getProperty(AppleFoundationModels.MODE_PROPERTY);
    try {
      System.setProperty(AppleFoundationModels.MODE_PROPERTY, "stub");

      try (AppleFoundationModelsClient client = AppleFoundationModels.create()) {
        assertThat(client.availability().available()).isTrue();
        assertThat(
                client
                    .generate(
                        AppleFoundationModelsRequest.builder("Summarize: apples are red.")
                            .instructions("Be concise.")
                            .maxOutputTokens(12)
                            .build())
                    .text())
            .contains("Stub summary");
      }
    } finally {
      if (previous == null) {
        System.clearProperty(AppleFoundationModels.MODE_PROPERTY);
      } else {
        System.setProperty(AppleFoundationModels.MODE_PROPERTY, previous);
      }
    }
  }
}
