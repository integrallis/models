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

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.catalog.DiscoveredModel;
import com.integrallis.models.api.catalog.ModelCatalogProvider;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceLoader;
import java.util.Set;

/**
 * Opens a model this machine's installed catalogs already hold, by its catalog id.
 *
 * <p>This is the step that was missing. {@link ModelCatalogProvider} has always reported what is
 * installed and what it measured, and the router has always consumed that — but opening one still
 * meant naming a file path the catalog already knew, in application code. So every caller that
 * wanted a catalog model wrote the resolution themselves, which is the opposite of the catalog
 * being authoritative.
 *
 * <p><b>The engine is this one.</b> Resolution ends at {@link PureJavaBackend#load(Path,
 * BackendConfiguration)}, so a catalog says where the weights are and the Java decoder runs them,
 * with the native shims picked up underneath exactly as they are for any other load. There is
 * deliberately no way to select a different engine here: an engine chosen at the call site is a
 * measurement seam, and the shipped path has to be the one the qualification measured.
 *
 * <p><b>Nothing is downloaded.</b> {@link ModelCatalogProvider#locate} reports only what is on
 * disk, so a catalog entry whose weights have not been fetched is a refusal with that fact in the
 * message, never a silent multi-gigabyte download on a lookup.
 */
public final class CatalogBackends {

  private CatalogBackends() {}

  /**
   * Opens the catalog model with this id, with no backend configuration overrides.
   *
   * @param id the catalog id, as {@link DiscoveredModel#id()} reports it
   * @return the opened backend, which the caller closes
   * @throws IllegalArgumentException when no installed catalog can open that id
   */
  public static PureJavaBackend open(String id) {
    return open(id, BackendConfiguration.empty());
  }

  /**
   * Opens the catalog model with this id.
   *
   * @param id the catalog id, as {@link DiscoveredModel#id()} reports it
   * @param configuration the backend configuration to open it with
   * @return the opened backend, which the caller closes
   * @throws IllegalArgumentException when no installed catalog can open that id
   */
  public static PureJavaBackend open(String id, BackendConfiguration configuration) {
    return open(ServiceLoader.load(ModelCatalogProvider.class), id, configuration);
  }

  /** The same, over a supplied provider set. */
  static PureJavaBackend open(
      Iterable<ModelCatalogProvider> providers, String id, BackendConfiguration configuration) {
    Resolution resolution = resolve(providers, id);
    return PureJavaBackend.load(resolution.path(), configuration);
  }

  /**
   * Where the installed catalogs say this model's weights are, without opening anything.
   *
   * @param id the catalog id
   * @return the path, or empty when no installed catalog holds that id on disk
   */
  public static Optional<Path> locate(String id) {
    return locate(ServiceLoader.load(ModelCatalogProvider.class), id);
  }

  /**
   * The same, over a supplied provider set; the seam the tests drive, as in {@code
   * CatalogDiscovery}.
   */
  static Optional<Path> locate(Iterable<ModelCatalogProvider> providers, String id) {
    requireText(id);
    for (ModelCatalogProvider provider : providers) {
      Optional<Path> located = locateQuietly(provider, id);
      if (located.isPresent()) {
        return located;
      }
    }
    return Optional.empty();
  }

  /**
   * Every catalog id that can be opened right now, in discovery order.
   *
   * <p>What a caller needs to put in front of a user who got the id wrong, and what the refusal
   * message below is built from. An id that appears in {@link ModelCatalogProvider#discover()} but
   * whose weights are not on disk is deliberately absent: this lists what can be opened, not what
   * could be after a download.
   *
   * @return the openable ids
   */
  public static Set<String> openable() {
    return openable(ServiceLoader.load(ModelCatalogProvider.class));
  }

  /** The same, over a supplied provider set. */
  static Set<String> openable(Iterable<ModelCatalogProvider> providers) {
    Set<String> ids = new LinkedHashSet<>();
    for (ModelCatalogProvider provider : providers) {
      for (DiscoveredModel model : discoverQuietly(provider)) {
        if (locateQuietly(provider, model.id()).isPresent()) {
          ids.add(model.id());
        }
      }
    }
    return ids;
  }

  record Resolution(String catalog, Path path) {}

  /**
   * Finds the one catalog that can open this id, and refuses rather than guessing.
   *
   * <p>Two catalogs holding the same id is not a tie to break. Whichever one a resolver picked, the
   * other would still be installed, and a later run with the providers loaded in a different order
   * would quietly open different weights under the same name. The refusal names both.
   */
  static Resolution resolve(String id) {
    return resolve(ServiceLoader.load(ModelCatalogProvider.class), id);
  }

  static Resolution resolve(Iterable<ModelCatalogProvider> providers, String id) {
    requireText(id);
    Map<String, Path> byCatalog = new LinkedHashMap<>();
    for (ModelCatalogProvider provider : providers) {
      locateQuietly(provider, id).ifPresent(located -> byCatalog.put(provider.name(), located));
    }

    if (byCatalog.size() == 1) {
      Map.Entry<String, Path> only = byCatalog.entrySet().iterator().next();
      return new Resolution(only.getKey(), only.getValue());
    }
    if (byCatalog.size() > 1) {
      throw new IllegalArgumentException(
          "catalog id '"
              + id
              + "' is held by more than one installed catalog "
              + byCatalog.keySet()
              + "; uninstall one or open the artifact by path");
    }
    // One refusal, listing what can actually be opened.
    //
    // An earlier version also tried to distinguish "this catalog knows the id, the weights just
    // are not here" by consulting discover(), and a cross-project end-to-end run on 2026-10-22
    // showed that was wrong twice over. ModelJarsCatalog reports only cached artifacts, so for the
    // one shipping catalog the branch could never fire; and the catalog where it did fire --
    // on-device platform intelligence, which describes a model with no path any process may open
    // -- would have been told to "fetch the artifact first", which is not a thing that can be
    // done. A distinction this layer cannot draw truthfully is better not drawn.
    throw new IllegalArgumentException(
        "no installed catalog can open '" + id + "'; openable ids are " + openable(providers));
  }

  /**
   * A provider that throws is skipped, not propagated.
   *
   * <p>{@link ModelCatalogProvider} already requires an absent catalog to be an empty list rather
   * than a failure, and the same reasoning covers one that is installed but broken: a caller asking
   * for a ModelJars id should not fail because some unrelated catalog on the classpath threw.
   */
  private static Optional<Path> locateQuietly(ModelCatalogProvider provider, String id) {
    try {
      Optional<Path> located = provider.locate(id);
      return located == null ? Optional.empty() : located;
    } catch (RuntimeException | LinkageError failure) {
      return Optional.empty();
    }
  }

  private static List<DiscoveredModel> discoverQuietly(ModelCatalogProvider provider) {
    try {
      List<DiscoveredModel> discovered = provider.discover();
      return discovered == null ? List.of() : discovered;
    } catch (RuntimeException | LinkageError failure) {
      return List.of();
    }
  }

  private static void requireText(String id) {
    if (id == null || id.isBlank()) {
      throw new IllegalArgumentException("id must not be blank");
    }
  }
}
