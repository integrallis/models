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

import com.integrallis.vectors.core.VectorUtil;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/**
 * Resolves the native worker count and reports obsolete deployment settings.
 *
 * <p>Earlier releases also exposed switches for quantized decode, grouped Q5_0, DeltaNet, load
 * warmup, and grouped attention. Published model profiles can still contain these settings, so they
 * are accepted and reported as ignored. They do not select an implementation in this release.
 *
 * <p>Routing belongs to the kernel provider and each model caller. A reported native capability
 * establishes availability; caller eligibility also depends on the operation, shape, and supported
 * hardware. Capability discovery is not a performance qualification. Removed settings must not
 * resurrect retired kernels or override those routing decisions.
 */
record NativeKernelSettings(int threadCount, List<String> ignoredRemovedSettings) {
  private static final String PROPERTY_PREFIX = "models.native.";
  private static final Set<String> SUPPORTED_SETTINGS =
      Set.of(NativeKernelLibrary.THREAD_COUNT_PROPERTY);

  /**
   * Settings that used to veto a Rust kernel and no longer exist.
   *
   * <p><b>These are accepted and ignored, never rejected.</b> Twenty-five published ModelJars
   * performance profiles recommend {@code models.native.quantizedDecode=true}, and a published
   * marker jar embeds its profile, so a version of this class that threw on them would stop those
   * models loading at all — and no catalogue edit could reach a marker already in Maven Central.
   *
   * <p>{@link #ignoredRemovedSettings()} reports which were seen so {@code RustFfmBackend} can
   * identify obsolete configuration in its diagnostics. The diagnostic does not imply that the
   * requested historical route still exists or is selected.
   */
  private static final Set<String> REMOVED_SETTINGS =
      Set.of(
          "models.native.quantizedDecode",
          "models.native.q5_0.grouped",
          "models.native.gatedDeltaNet",
          "models.native.loadWarmup",
          "models.native.groupedAttention");

  NativeKernelSettings {
    if (threadCount < 1 || threadCount > 256) {
      throw new IllegalArgumentException("threadCount must be between 1 and 256: " + threadCount);
    }
    ignoredRemovedSettings = List.copyOf(ignoredRemovedSettings);
  }

  /** Convenience for the common case of no removed settings in play. */
  NativeKernelSettings(int threadCount) {
    this(threadCount, List.of());
  }

  static NativeKernelSettings fromSystemProperties(Map<String, String> recommendations) {
    Properties systemProperties = System.getProperties();
    Map<String, String> deployment = new LinkedHashMap<>();
    List<String> ignored = new ArrayList<>();
    synchronized (systemProperties) {
      REMOVED_SETTINGS.stream()
          .sorted()
          .filter(property -> systemProperties.getProperty(property) != null)
          .forEach(property -> ignored.add(property + " (deployment)"));
      SUPPORTED_SETTINGS.stream()
          .sorted()
          .forEach(
              property -> {
                String value = systemProperties.getProperty(property);
                if (value != null) {
                  deployment.put(property, value);
                }
              });
    }
    // The same number the vectors GGUF executor sizes itself to, not an independent reading of the
    // machine. Two pools each taking availableProcessors() put twice the machine's threads on it,
    // and on a hybrid part availableProcessors() counts efficiency cores as equals: MEASURED on a
    // 32-logical i9-14900HX (8 performance cores, 16 efficiency), one typed decision cost 2530 ms
    // at that default and 877 ms at the eight threads this resolves to. A barrier makes a dispatch
    // cost whatever its slowest member cost, so an efficiency-core thread held up all thirty-two.
    return resolve(recommendations, Map.copyOf(deployment), sharedComputeThreads())
        .withIgnored(ignored);
  }

  private NativeKernelSettings withIgnored(List<String> additional) {
    if (additional.isEmpty()) {
      return this;
    }
    List<String> combined = new ArrayList<>(ignoredRemovedSettings);
    combined.addAll(additional);
    return new NativeKernelSettings(threadCount, combined);
  }

  static NativeKernelSettings resolve(
      Map<String, String> recommendations, Map<String, String> deployment, int defaultThreadCount) {
    Objects.requireNonNull(recommendations, "recommendations");
    Objects.requireNonNull(deployment, "deployment");
    validateSettings(recommendations, "recommendation");
    validateSettings(deployment, "deployment setting");
    List<String> ignored = new ArrayList<>();
    REMOVED_SETTINGS.stream()
        .sorted()
        .forEach(
            property -> {
              if (recommendations.containsKey(property)) {
                ignored.add(property + " (profile)");
              }
              if (deployment.containsKey(property)) {
                ignored.add(property + " (deployment)");
              }
            });
    return new NativeKernelSettings(
        threadCount(
            configured(NativeKernelLibrary.THREAD_COUNT_PROPERTY, deployment, recommendations),
            defaultThreadCount),
        ignored);
  }

  /**
   * The thread count the vectors GGUF executor sizes itself to, or this JVM's processor count if it
   * cannot say.
   *
   * <p>The fallback is not decoration. A thread count is a number this class validates to be
   * between 1 and 256, so it must not accept whatever an external provider hands back: a provider
   * that cannot answer, or one replaced wholesale -- a test that calls {@code
   * mockStatic(VectorUtil.class)} leaves every unstubbed static returning 0 -- would otherwise turn
   * a missing answer into a settings failure far from its cause. Guarding here keeps the dependency
   * one-directional: this class asks for a recommendation and remains responsible for the number it
   * uses.
   */
  private static int sharedComputeThreads() {
    int shared = VectorUtil.computeThreads();
    return shared > 0 ? shared : Math.max(1, Runtime.getRuntime().availableProcessors());
  }

  private static String configured(
      String property, Map<String, String> deployment, Map<String, String> recommendations) {
    String value = deployment.get(property);
    return value != null ? value : recommendations.get(property);
  }

  private static int threadCount(String value, int defaultThreadCount) {
    if (value == null || value.isBlank()) {
      return checkedThreadCount(defaultThreadCount, Integer.toString(defaultThreadCount));
    }
    try {
      return checkedThreadCount(Integer.parseInt(value), value);
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException(
          NativeKernelLibrary.THREAD_COUNT_PROPERTY + " must be an integer: " + value, failure);
    }
  }

  private static int checkedThreadCount(int threadCount, String value) {
    if (threadCount < 1 || threadCount > 256) {
      throw new IllegalArgumentException(
          NativeKernelLibrary.THREAD_COUNT_PROPERTY + " must be between 1 and 256: " + value);
    }
    return threadCount;
  }

  private static void validateSettings(Map<String, String> settings, String source) {
    settings.keySet().stream()
        .filter(key -> key.startsWith(PROPERTY_PREFIX))
        .filter(key -> !SUPPORTED_SETTINGS.contains(key))
        // A removed setting is tolerated: see REMOVED_SETTINGS. Only a key that never existed is an
        // error, because that is a typo or a stale name the caller needs told about.
        .filter(key -> !REMOVED_SETTINGS.contains(key))
        .findFirst()
        .ifPresent(
            key -> {
              throw new IllegalArgumentException("Unsupported native " + source + ": " + key);
            });
  }
}
