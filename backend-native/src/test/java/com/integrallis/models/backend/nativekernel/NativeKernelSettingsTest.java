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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * There is one native route, and thread count is the only thing a deployment still chooses.
 *
 * <p>{@code models.native.quantizedDecode}, {@code models.native.q5_0.grouped}, {@code
 * models.native.gatedDeltaNet} and {@code models.native.loadWarmup} were removed. Each vetoed a
 * Rust kernel the shim had already reported a capability for, and each defaulted to off when no
 * deployment property and no exactly-matching ModelJar profile supplied it — which was the common
 * case, since every published profile pins an exact {@code cpu-model}, {@code processors} and
 * usually an exact {@code vm-version}. Decode on the Java path measured 6.2x slower on the same
 * host and harness, so the slow route was not worth keeping as a choice.
 *
 * <p>These tests fail if a veto is reintroduced as a recognised setting.
 */
class NativeKernelSettingsTest {

  @Test
  @DisplayName("thread count is the only native setting, and it follows the host when unset")
  void threadCountIsTheOnlySetting() {
    NativeKernelSettings settings = NativeKernelSettings.resolve(Map.of(), Map.of(), 8);

    assertThat(settings.threadCount()).isEqualTo(8);
    assertThat(NativeKernelSettings.class.getRecordComponents())
        .as("thread count plus the ignored-settings record; a new route must be deliberate")
        .hasSize(2);
  }

  @Test
  @DisplayName("a deployment property overrides a profile recommendation")
  void deploymentOverridesRecommendation() {
    NativeKernelSettings settings =
        NativeKernelSettings.resolve(
            Map.of(NativeKernelLibrary.THREAD_COUNT_PROPERTY, "6"),
            Map.of(NativeKernelLibrary.THREAD_COUNT_PROPERTY, "12"),
            8);

    assertThat(settings.threadCount()).isEqualTo(12);
  }

  @Test
  @DisplayName("a published profile naming a removed setting still loads, and says it was ignored")
  void removedSettingsAreToleratedAndRecorded() {
    // Twenty-five published ModelJars profiles recommend models.native.quantizedDecode=true, and a
    // marker jar embeds its profile, so rejecting these would stop those models loading -- and no
    // catalogue edit can reach a marker already on Maven Central.
    for (String removed :
        new String[] {
          "models.native.quantizedDecode",
          "models.native.q5_0.grouped",
          "models.native.gatedDeltaNet",
          "models.native.loadWarmup",
          "models.native.groupedAttention"
        }) {
      NativeKernelSettings fromProfile =
          NativeKernelSettings.resolve(Map.of(removed, "true"), Map.of(), 8);
      assertThat(fromProfile.threadCount()).isEqualTo(8);
      assertThat(fromProfile.ignoredRemovedSettings())
          .as("profile recommendation %s must be recorded, not dropped", removed)
          .containsExactly(removed + " (profile)");

      NativeKernelSettings fromDeployment =
          NativeKernelSettings.resolve(Map.of(), Map.of(removed, "true"), 8);
      assertThat(fromDeployment.ignoredRemovedSettings())
          .containsExactly(removed + " (deployment)");
    }
  }

  @Test
  @DisplayName("a setting that never existed is still an error")
  void unknownSettingsAreRejected() {
    assertThatThrownBy(
            () -> NativeKernelSettings.resolve(Map.of(), Map.of("models.native.nonsense", "1"), 8))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.native.nonsense");
  }

  @Test
  @DisplayName("no removed setting means nothing is recorded")
  void cleanConfigurationRecordsNothing() {
    assertThat(NativeKernelSettings.resolve(Map.of(), Map.of(), 8).ignoredRemovedSettings())
        .isEmpty();
  }

  @Test
  @DisplayName("an out-of-range thread count is rejected")
  void threadCountIsValidated() {
    assertThatThrownBy(
            () ->
                NativeKernelSettings.resolve(
                    Map.of(), Map.of(NativeKernelLibrary.THREAD_COUNT_PROPERTY, "0"), 8))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                NativeKernelSettings.resolve(
                    Map.of(), Map.of(NativeKernelLibrary.THREAD_COUNT_PROPERTY, "nope"), 8))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
