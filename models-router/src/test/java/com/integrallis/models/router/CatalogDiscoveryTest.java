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
package com.integrallis.models.router;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.catalog.DiscoveredModel;
import com.integrallis.models.api.catalog.ModelCatalogProvider;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Contract tests for turning installed catalogs into routing candidates. */
@Tag("unit")
class CatalogDiscoveryTest {

  private static DiscoveredModel model(String id, DiscoveredModel.Performance performance) {
    return new DiscoveredModel(
        id, true, Set.of("code"), 32768, 0.0, 0.0, performance, Map.of("code", 0.82), 0.99);
  }

  private record FakeCatalog(String name, List<DiscoveredModel> models)
      implements ModelCatalogProvider {
    @Override
    public List<DiscoveredModel> discover() {
      return models;
    }
  }

  @Test
  void mapsMeasuredModelsIntoCandidates() {
    List<ModelCandidate> candidates =
        CatalogDiscovery.discover(
            List.of(
                new FakeCatalog(
                    "fake",
                    List.of(model("qwen-coder", new DiscoveredModel.Performance(180, 42.0))))));

    assertThat(candidates).hasSize(1);
    ModelCandidate candidate = candidates.get(0);
    assertThat(candidate.id()).isEqualTo("qwen-coder");
    assertThat(candidate.local()).isTrue();
    assertThat(candidate.timeToFirstTokenMillis()).isEqualTo(180);
    assertThat(candidate.tokensPerSecond()).isEqualTo(42.0);
    assertThat(candidate.tags()).containsExactly("code");
    assertThat(candidate.qualityFor("code")).isEqualTo(0.82);
  }

  @Test
  void skipsModelsWithNoProfileForThisHardware() {
    // Defaulting instead would let the router prefer this model on invented latency, confidently
    // and silently. Absent is the honest answer, and CatalogDiscovery says so at WARNING.
    List<ModelCandidate> candidates =
        CatalogDiscovery.discover(
            List.of(new FakeCatalog("fake", List.of(model("unprofiled", null)))));

    assertThat(candidates).isEmpty();
  }

  @Test
  void survivesACatalogThatThrows() {
    ModelCatalogProvider broken =
        new ModelCatalogProvider() {
          @Override
          public String name() {
            return "broken";
          }

          @Override
          public List<DiscoveredModel> discover() {
            throw new IllegalStateException("catalog file is corrupt");
          }
        };
    ModelCatalogProvider working =
        new FakeCatalog(
            "working", List.of(model("survivor", new DiscoveredModel.Performance(200, 30.0))));

    // Two catalogs installed and one broken: the caller still deserves the models from the other.
    assertThat(CatalogDiscovery.discover(List.of(broken, working)))
        .extracting(ModelCandidate::id)
        .containsExactly("survivor");
  }

  @Test
  void returnsEmptyWhenNoCatalogIsInstalled() {
    // Not an error. Routing between hosted models only is a legitimate configuration, so this
    // logs how to install a catalog and carries on.
    assertThat(CatalogDiscovery.discover(List.of())).isEmpty();
  }

  private record OptInCatalog(String name, List<DiscoveredModel> models)
      implements ModelCatalogProvider {
    @Override
    public List<DiscoveredModel> discover() {
      return models;
    }

    @Override
    public boolean requiresOptIn() {
      return true;
    }
  }

  @Test
  void leavesOnDeviceIntelligenceOutUnlessAskedFor() {
    OptInCatalog apple =
        new OptInCatalog(
            "apple-foundation-models",
            List.of(model("apple-foundation-model", new DiscoveredModel.Performance(90, 20.0))));

    // Default off: hardware that happens to be present must not change which model the same code
    // selects between a developer's Mac and production.
    assertThat(CatalogDiscovery.discover(List.of(apple), false)).isEmpty();
    assertThat(CatalogDiscovery.discover(List.of(apple), true))
        .extracting(ModelCandidate::id)
        .containsExactly("apple-foundation-model");
  }

  @Test
  void optingInDoesNotDisturbOrdinaryCatalogs() {
    ModelCatalogProvider installed =
        new FakeCatalog("fake", List.of(model("qwen", new DiscoveredModel.Performance(180, 42.0))));
    OptInCatalog apple =
        new OptInCatalog(
            "apple",
            List.of(model("apple-foundation-model", new DiscoveredModel.Performance(90, 20.0))));

    assertThat(CatalogDiscovery.discover(List.of(installed, apple), false))
        .extracting(ModelCandidate::id)
        .containsExactly("qwen");
    assertThat(CatalogDiscovery.discover(List.of(installed, apple), true))
        .extracting(ModelCandidate::id)
        .containsExactly("qwen", "apple-foundation-model");
  }

  @Test
  void discoveredModelRejectsFiguresThatCannotBeMeasurements() {
    assertThat(
            org.assertj.core.api.Assertions.catchThrowable(
                () -> new DiscoveredModel.Performance(-1, 10.0)))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(
            org.assertj.core.api.Assertions.catchThrowable(
                () -> new DiscoveredModel.Performance(10, 0.0)))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
