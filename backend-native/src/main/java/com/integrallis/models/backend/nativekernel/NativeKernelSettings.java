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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/**
 * The one native setting a deployment can still choose: how many worker threads the shim runs.
 *
 * <p>There used to be four more — {@code quantizedDecode}, {@code q5_0.grouped}, {@code
 * gatedDeltaNet} and {@code loadWarmup} — each of which vetoed a Rust kernel the shim had already
 * reported a capability for, and each of which defaulted to <em>off</em> when no deployment
 * property and no matching ModelJar profile supplied it. That was the common case: every published
 * performance profile selects on an exact {@code os}, {@code architecture}, {@code cpu-model},
 * {@code processors} and usually an exact {@code vm-version}, so any host outside the two EC2 CPU
 * models we qualify on matched nothing and ran the slow route. Routing decode to the Java path cost
 * 6.2x on the same host and harness — 2.52 against 15.72 tok/s, OFFLINE against USABLE
 * (`benchmark-results/2026-09-23-prefill-budget/NOTES.md`) — while every certified qualification
 * run had it forced on, so the published tiers described a path users did not get.
 *
 * <p>They are gone rather than flipped. The shim serves whatever it reports a capability for, and
 * the Java path remains only where there is no capability, which is a real fallback and not a
 * choice. {@code loadWarmup} is gone outright: its experiment returned VOID, so it bought no
 * measured speed while costing load time against the readiness gate.
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
   * <p>Ignoring them is also the truthful answer rather than a concession. Each named a faster
   * route that the shim now takes unconditionally, so a profile asking for it gets exactly what it
   * asked for. {@link #ignoredRemovedSettings()} reports which were seen so {@code RustFfmBackend}
   * can record them in its diagnostics: a caller reading the report finds the setting named and
   * marked ignored, instead of silently believing it still does something.
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
    return resolve(
            recommendations, Map.copyOf(deployment), Runtime.getRuntime().availableProcessors())
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
