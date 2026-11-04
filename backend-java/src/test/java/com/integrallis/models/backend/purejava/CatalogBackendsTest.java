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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.catalog.DiscoveredModel;
import com.integrallis.models.api.catalog.ModelCatalogProvider;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Resolving a catalog id to an artifact, which is the step that made the catalog authoritative.
 *
 * <p>No model is opened here. {@code open} ends at {@link PureJavaBackend#load}, which needs real
 * weights; what these tests pin is everything before that — which catalog wins, what happens when
 * two do, and the difference between an id nobody knows and an id whose weights are simply not on
 * this machine. That last distinction is the one a caller acts on.
 */
@Tag("unit")
class CatalogBackendsTest {

  @Test
  void resolvesAnIdTheOneInstalledCatalogHoldsOnDisk() {
    Path artifact = Path.of("/models/qwen3-1.7b-q8_0.gguf");
    Iterable<ModelCatalogProvider> providers =
        List.of(catalog("modeljars", Map.of("qwen3_1_7b_q8_0", artifact)));

    assertThat(CatalogBackends.locate(providers, "qwen3_1_7b_q8_0")).contains(artifact);
    assertThat(CatalogBackends.resolve(providers, "qwen3_1_7b_q8_0"))
        .isEqualTo(new CatalogBackends.Resolution("modeljars", artifact));
  }

  @Test
  void refusesAnIdTwoInstalledCatalogsBothHold() {
    // Not a tie to break: whichever won, the other would still be installed, and a later run with
    // the providers loaded in a different order would open different weights under one name.
    Iterable<ModelCatalogProvider> providers =
        List.of(
            catalog("modeljars", Map.of("shared_id", Path.of("/a/model.gguf"))),
            catalog("vendor-pack", Map.of("shared_id", Path.of("/b/model.gguf"))));

    assertThatThrownBy(() -> CatalogBackends.resolve(providers, "shared_id"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("more than one installed catalog")
        .hasMessageContaining("modeljars")
        .hasMessageContaining("vendor-pack");
  }

  @Test
  void refusesAnIdItCannotOpenWithTheListOfIdsItCan() {
    // One refusal for everything unopenable, and deliberately so. A previous version told a caller
    // whose id was described but unlocatable to "fetch the artifact first"; that branch was
    // unreachable for ModelJars, which reports only cached artifacts, and wrong for on-device
    // platform intelligence, which describes a model that cannot be fetched at all.
    Iterable<ModelCatalogProvider> providers =
        List.of(
            catalog("modeljars", Map.of("held", Path.of("/models/held.gguf"))),
            describesButDoesNotHold("platform-intelligence", "on_device_only"));

    for (String unopenable : List.of("on_device_only", "never_heard_of_it")) {
      assertThatThrownBy(() -> CatalogBackends.resolve(providers, unopenable))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("no installed catalog can open")
          .hasMessageContaining(unopenable)
          .hasMessageContaining("held")
          // Never advice that cannot be followed.
          .hasMessageNotContaining("fetch");
    }
  }

  @Test
  void listsTheOpenableIdsOfTheProvidersItWasGivenNotOfTheWholeMachine() {
    // The refusal is built from the supplied providers, so a caller reading it sees the set that
    // was actually consulted. Reading the installed catalogs here instead would make the message
    // describe a different machine than the lookup did.
    Iterable<ModelCatalogProvider> providers =
        List.of(catalog("modeljars", Map.of("only_this_one", Path.of("/models/one.gguf"))));

    assertThatThrownBy(() -> CatalogBackends.resolve(providers, "missing"))
        .hasMessageContaining("[only_this_one]");
  }

  @Test
  void listsOnlyWhatCanBeOpenedNowNotEverythingDescribed() {
    Iterable<ModelCatalogProvider> providers =
        List.of(
            catalog("modeljars", Map.of("held", Path.of("/models/held.gguf"))),
            describesButDoesNotHold("vendor-pack", "not-fetched"));

    assertThat(CatalogBackends.openable(providers)).containsExactly("held");
  }

  @Test
  void skipsACatalogThatThrowsInsteadOfFailingTheLookup() {
    // An unrelated broken catalog on the classpath must not stop a ModelJars id resolving. The
    // SPI already requires an absent catalog to be an empty list rather than a failure.
    Path artifact = Path.of("/models/good.gguf");
    Iterable<ModelCatalogProvider> providers =
        List.of(exploding("broken"), catalog("modeljars", Map.of("good_id", artifact)));

    assertThat(CatalogBackends.locate(providers, "good_id")).contains(artifact);
    assertThat(CatalogBackends.openable(providers)).containsExactly("good_id");
  }

  @Test
  void treatsACatalogThatReturnsNullAsOneThatHoldsNothing() {
    Iterable<ModelCatalogProvider> providers = List.of(returnsNull("sloppy"));

    assertThat(CatalogBackends.locate(providers, "anything")).isEmpty();
    assertThat(CatalogBackends.openable(providers)).isEmpty();
  }

  @Test
  void refusesABlankId() {
    assertThatThrownBy(() -> CatalogBackends.locate(List.of(), " "))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("must not be blank");
  }

  @Test
  void aCatalogWrittenBeforeLocateExistedStillCompilesAndLocatesNothing() {
    // The default on the SPI is what keeps an older provider meaning what it meant: it describes
    // models and locates none. Asserted rather than assumed, since the default is the whole
    // compatibility story for this addition.
    ModelCatalogProvider legacy =
        new ModelCatalogProvider() {
          @Override
          public String name() {
            return "legacy";
          }

          @Override
          public List<DiscoveredModel> discover() {
            return List.of(model("legacy_model"));
          }
        };

    assertThat(legacy.locate("legacy_model")).isEmpty();
    assertThat(CatalogBackends.openable(List.of(legacy))).isEmpty();
    assertThatThrownBy(() -> CatalogBackends.resolve(List.of(legacy), "legacy_model"))
        .hasMessageContaining("no installed catalog can open");
  }

  private static ModelCatalogProvider catalog(String name, Map<String, Path> held) {
    return new ModelCatalogProvider() {
      @Override
      public String name() {
        return name;
      }

      @Override
      public List<DiscoveredModel> discover() {
        return held.keySet().stream().map(CatalogBackendsTest::model).toList();
      }

      @Override
      public Optional<Path> locate(String id) {
        return Optional.ofNullable(held.get(id));
      }
    };
  }

  private static ModelCatalogProvider describesButDoesNotHold(String name, String id) {
    return new ModelCatalogProvider() {
      @Override
      public String name() {
        return name;
      }

      @Override
      public List<DiscoveredModel> discover() {
        return List.of(model(id));
      }

      @Override
      public Optional<Path> locate(String requested) {
        return Optional.empty();
      }
    };
  }

  private static ModelCatalogProvider exploding(String name) {
    return new ModelCatalogProvider() {
      @Override
      public String name() {
        return name;
      }

      @Override
      public List<DiscoveredModel> discover() {
        throw new IllegalStateException("catalog is broken");
      }

      @Override
      public Optional<Path> locate(String id) {
        throw new IllegalStateException("catalog is broken");
      }
    };
  }

  private static ModelCatalogProvider returnsNull(String name) {
    return new ModelCatalogProvider() {
      @Override
      public String name() {
        return name;
      }

      @Override
      public List<DiscoveredModel> discover() {
        return null;
      }

      @Override
      public Optional<Path> locate(String id) {
        return null;
      }
    };
  }

  private static DiscoveredModel model(String id) {
    return new DiscoveredModel(
        id,
        true,
        Set.of("chat"),
        Set.of("chat"),
        32768,
        0.0,
        0.0,
        1024L,
        new DiscoveredModel.Performance(100L, 20.0),
        Map.of(),
        0.99);
  }
}
