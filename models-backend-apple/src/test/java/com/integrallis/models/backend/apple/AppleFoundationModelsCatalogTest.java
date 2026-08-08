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

import com.integrallis.models.api.catalog.DiscoveredModel;
import com.integrallis.models.api.catalog.ModelCatalogProvider;
import java.util.List;
import java.util.ServiceLoader;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Tests for the on-device catalog.
 *
 * <p>These run on every platform, which is the point: the common case is a machine with no Apple
 * Intelligence at all, and reporting nothing there must be silent and cheap rather than an error
 * that surfaces in someone's build.
 */
@Tag("unit")
class AppleFoundationModelsCatalogTest {

  @Test
  void isRegisteredForServiceLoaderDiscovery() {
    // The catalog is useless unless discovery can find it, and a missing services file is exactly
    // the kind of packaging mistake nothing else would catch.
    assertThat(ServiceLoader.load(ModelCatalogProvider.class).stream())
        .anyMatch(p -> p.type().equals(AppleFoundationModelsCatalog.class));
  }

  @Test
  void requiresOptInBecauseHardwareIsNotAChoice() {
    assertThat(new AppleFoundationModelsCatalog().requiresOptIn()).isTrue();
  }

  @Test
  void namesItselfForDiagnostics() {
    assertThat(new AppleFoundationModelsCatalog().name()).isEqualTo("apple-foundation-models");
  }

  @Test
  void reportsNothingWhereThereIsNoOnDeviceModel() {
    // On CI, on Linux, and on an Intel Mac this is the whole behaviour: an empty list, no
    // exception, and nothing written to the caller's logs.
    List<DiscoveredModel> discovered = new AppleFoundationModelsCatalog().discover();

    assertThat(discovered).isNotNull();
    if (!isAppleSiliconMac()) {
      assertThat(discovered).isEmpty();
    }
  }

  @Test
  void anythingItDoesReportCarriesRealMeasurements() {
    // Skipped everywhere but a capable Mac. Asserted rather than assumed because a model reported
    // without a Performance would be silently dropped by CatalogDiscovery, making the opt-in a
    // no-op that looks like it worked.
    for (DiscoveredModel model : new AppleFoundationModelsCatalog().discover()) {
      assertThat(model.id()).isEqualTo(AppleFoundationModelsCatalog.MODEL_ID);
      assertThat(model.local()).isTrue();
      assertThat(model.costPerMillionInputTokens()).isZero();
      assertThat(model.performance()).isNotNull();
      assertThat(model.performance().tokensPerSecond()).isPositive();
    }
  }

  private static boolean isAppleSiliconMac() {
    String os = System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT);
    String arch = System.getProperty("os.arch", "").toLowerCase(java.util.Locale.ROOT);
    return os.contains("mac") && (arch.equals("aarch64") || arch.equals("arm64"));
  }
}
