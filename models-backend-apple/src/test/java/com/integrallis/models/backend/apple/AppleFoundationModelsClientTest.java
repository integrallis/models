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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class AppleFoundationModelsClientTest {

  @Test
  void unavailableClientReportsReasonAndRejectsGeneration() {
    AppleFoundationModelsClient client =
        AppleFoundationModelsClient.unavailable("Apple Foundation Models unavailable");

    assertThat(client.availability().available()).isFalse();
    assertThat(client.availability().reason()).contains("unavailable");
    assertThatThrownBy(() -> client.generate("Summarize this"))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("Apple Foundation Models unavailable");
  }

  @Test
  void generateValidatesPrompt() {
    AppleFoundationModelsClient client = AppleFoundationModelsClient.of(new EchoBridge());

    assertThatThrownBy(() -> client.generate((String) null))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> client.generate("   "))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("prompt");
  }

  @Test
  void generateDelegatesToBridge() {
    AppleFoundationModelsClient client = AppleFoundationModelsClient.of(new EchoBridge());

    AppleFoundationModelsResponse response =
        client.generate(
            AppleFoundationModelsRequest.builder("Write a haiku")
                .instructions("Be terse")
                .maxOutputTokens(32)
                .build());

    assertThat(response.text()).isEqualTo("Be terse :: Write a haiku :: 32");
  }

  @Test
  void facadeReturnsUnavailableOnUnsupportedPlatform() {
    AppleFoundationModelsClient client =
        AppleFoundationModels.create(
            new ApplePlatform("Mac OS X", "x86_64"), NativeLibraryLocator.empty());

    assertThat(client.availability().supported()).isFalse();
    assertThat(client.availability().reason()).contains("Apple silicon");
  }

  private static final class EchoBridge implements AppleFoundationModelsBridge {

    @Override
    public AppleFoundationModelsAvailability availability() {
      return AppleFoundationModelsAvailability.availableNow();
    }

    @Override
    public AppleFoundationModelsResponse generate(AppleFoundationModelsRequest request) {
      return new AppleFoundationModelsResponse(
          request.instructions() + " :: " + request.prompt() + " :: " + request.maxOutputTokens());
    }

    @Override
    public void close() {}
  }
}
