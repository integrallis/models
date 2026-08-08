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

import com.integrallis.models.api.catalog.DiscoveredModel;
import com.integrallis.models.api.catalog.ModelCatalogProvider;
import java.lang.System.Logger;
import java.lang.System.Logger.Level;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.ServiceLoader;

/**
 * Turns installed catalogs into routing candidates.
 *
 * <p>Exists so that using the router does not begin with writing out every model's price, latency
 * and per-task quality by hand — figures a caller has no way to know and would have to invent.
 *
 * <p>Nothing here throws for an absent catalog. A caller routing only between hosted models has no
 * catalog and is not in error; they get an empty list and a message saying how to install one.
 */
public final class CatalogDiscovery {

  private static final Logger LOG = System.getLogger(CatalogDiscovery.class.getName());

  private CatalogDiscovery() {}

  /**
   * Finds every model the installed catalogs report.
   *
   * @return candidates in catalog order, or empty when no catalog is installed
   */
  public static List<ModelCandidate> discover() {
    return discover(false);
  }

  /**
   * Finds every model the installed catalogs report.
   *
   * @param includeOnDeviceIntelligence whether to include catalogs that require opting in, which
   *     today means on-device platform intelligence such as Apple Foundation Models
   * @return candidates in catalog order, or empty when no catalog is installed
   */
  public static List<ModelCandidate> discover(boolean includeOnDeviceIntelligence) {
    return discover(ServiceLoader.load(ModelCatalogProvider.class), includeOnDeviceIntelligence);
  }

  /**
   * Finds every model the supplied catalogs report.
   *
   * @param providers the catalogs to read
   * @return candidates in catalog order
   */
  static List<ModelCandidate> discover(Iterable<ModelCatalogProvider> providers) {
    return discover(providers, false);
  }

  /**
   * Finds every model the supplied catalogs report.
   *
   * @param providers the catalogs to read
   * @param includeOnDeviceIntelligence whether opt-in catalogs contribute
   * @return candidates in catalog order
   */
  static List<ModelCandidate> discover(
      Iterable<ModelCatalogProvider> providers, boolean includeOnDeviceIntelligence) {
    Objects.requireNonNull(providers, "providers");
    // Collected first so estimates can be calibrated against everything this machine has actually
    // measured, whichever catalog reported it.
    List<DiscoveredModel> discoveredModels = new ArrayList<>();
    int catalogs = 0;

    for (ModelCatalogProvider provider : providers) {
      if (provider.requiresOptIn() && !includeOnDeviceIntelligence) {
        // Silently present hardware is exactly what should not be silently routed to: the same
        // code would otherwise pick a different model on a Mac than in production.
        LOG.log(
            Level.DEBUG,
            "skipping opt-in catalog {0}; pass discoverLocal(true) to include on-device"
                + " intelligence",
            provider.name());
        continue;
      }
      catalogs++;
      List<DiscoveredModel> discovered;
      try {
        discovered = provider.discover();
      } catch (RuntimeException e) {
        // One broken catalog must not take down discovery: a caller with two installed still
        // deserves the models from the working one.
        LOG.log(Level.WARNING, "catalog " + provider.name() + " failed to enumerate models", e);
        continue;
      }
      if (discovered == null) {
        continue;
      }
      discoveredModels.addAll(discovered);
    }

    PerformanceEstimator estimator = PerformanceEstimator.from(discoveredModels);
    List<ModelCandidate> candidates = new ArrayList<>();
    List<String> estimated = new ArrayList<>();
    for (DiscoveredModel model : discoveredModels) {
      DiscoveredModel.Performance performance = model.performance();
      if (performance == null) {
        // Estimated rather than skipped: a model absent from the fleet cannot be chosen at all, and
        // the usual reason it lacks a profile is that it was installed recently.
        performance = estimator.estimate(model);
        estimated.add(model.id());
      }
      candidates.add(toCandidate(model, performance));
    }
    if (!estimated.isEmpty()) {
      // Named, because an estimate that scores like a measurement is the thing worth knowing about.
      LOG.log(
          Level.INFO,
          "estimated throughput for {0} model(s) with no profile for this hardware: {1}."
              + " Estimates are {2}; benchmark them for measured figures.",
          estimated.size(),
          String.join(", ", estimated),
          estimator.calibrated()
              ? "calibrated against models measured here"
              : "conservative defaults, since nothing on this machine has been measured");
    }

    if (catalogs == 0) {
      LOG.log(
          Level.INFO,
          "No ModelJars catalog found locally, install with"
              + " 'implementation(\"org.modeljars:modeljars\")' to discover installed models"
              + " automatically, or register candidates explicitly with"
              + " ModelCandidate.builder(id).");
    } else if (candidates.isEmpty()) {
      LOG.log(Level.INFO, "{0} catalog(s) found but none reported any models.", catalogs);
    }
    return List.copyOf(candidates);
  }

  private static ModelCandidate toCandidate(
      DiscoveredModel model, DiscoveredModel.Performance performance) {
    return ModelCandidate.builder(model.id())
        .local(model.local())
        .tags(model.tags())
        .costPerMillionTokens(model.costPerMillionInputTokens(), model.costPerMillionOutputTokens())
        .timeToFirstTokenMillis(performance.timeToFirstTokenMillis())
        .tokensPerSecond(performance.tokensPerSecond())
        .contextWindow(model.contextWindow())
        .quality(model.quality())
        .successRate(model.successRate())
        .build();
  }
}
