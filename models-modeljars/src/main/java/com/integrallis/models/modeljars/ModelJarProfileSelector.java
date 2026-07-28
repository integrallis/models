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
package com.integrallis.models.modeljars;

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarException;
import org.modeljars.ModelPerformanceProfile;
import org.modeljars.ModelPerformanceProfileRegistry;

/** Converts exact ModelJars performance evidence into registry-neutral Models configuration. */
final class ModelJarProfileSelector {

  private static final Set<String> RESTART_SELECTOR_KEYS =
      Set.of("compiler", "java-feature", "java-version", "vm-name", "vm-vendor", "vm-version");

  private ModelJarProfileSelector() {}

  static BackendConfiguration select(
      ModelJarDescriptor descriptor,
      ModelPerformanceProfileRegistry registry,
      Map<String, String> runtime,
      List<String> inputArguments) {
    return select(descriptor, registry, runtime, inputArguments, "pure-java");
  }

  static BackendConfiguration select(
      ModelJarDescriptor descriptor,
      ModelPerformanceProfileRegistry registry,
      Map<String, String> runtime,
      List<String> inputArguments,
      String backend) {
    Objects.requireNonNull(descriptor, "descriptor");
    Objects.requireNonNull(registry, "registry");
    return select(descriptor, registry.profilesFor(descriptor), runtime, inputArguments, backend);
  }

  static BackendConfiguration select(
      ModelJarDescriptor descriptor,
      List<ModelPerformanceProfile> profiles,
      Map<String, String> runtime,
      List<String> inputArguments) {
    return select(descriptor, profiles, runtime, inputArguments, "pure-java");
  }

  static BackendConfiguration select(
      ModelJarDescriptor descriptor,
      List<ModelPerformanceProfile> profiles,
      Map<String, String> runtime,
      List<String> inputArguments,
      String backend) {
    Objects.requireNonNull(descriptor, "descriptor");
    Objects.requireNonNull(profiles, "profiles");
    Objects.requireNonNull(runtime, "runtime");
    Objects.requireNonNull(inputArguments, "inputArguments");
    if (backend == null || backend.isBlank()) {
      throw new IllegalArgumentException("backend must not be blank");
    }

    Map<String, String> environment = new LinkedHashMap<>();
    environment.put("modeljar-alias", descriptor.alias());
    environment.put("modeljar-coordinate", descriptor.markerCoordinate().toString());
    descriptor.sha256().ifPresent(value -> environment.put("modeljar-sha256", value));

    Map<String, String> recommendations = new LinkedHashMap<>();
    List<OptimizationDecision> decisions = new ArrayList<>();
    profiles.stream()
        .filter(profile -> backend.equals(profile.backend()))
        .forEach(
            profile -> {
              Evaluation evaluation = evaluate(profile, runtime, inputArguments);
              decisions.add(evaluation.decision());
              if (evaluation.applies()) {
                mergeRecommendations(recommendations, profile);
              }
            });
    return new BackendConfiguration(environment, recommendations, decisions);
  }

  private static Evaluation evaluate(
      ModelPerformanceProfile profile, Map<String, String> runtime, List<String> inputArguments) {
    List<String> mismatches = selectorMismatches(profile.runtimeSelector(), runtime);
    boolean platformMismatch =
        profile.runtimeSelector().keySet().stream()
            .filter(key -> !RESTART_SELECTOR_KEYS.contains(key))
            .anyMatch(key -> mismatch(profile.runtimeSelector().get(key), runtime.get(key)));
    List<String> missingArguments =
        profile
            .javaLaunch()
            .map(launch -> launch.missingArguments(inputArguments))
            .orElse(List.of());

    OptimizationStatus status;
    String reason;
    if (!profile.safeForAutomaticSelection()) {
      status = OptimizationStatus.UNSUPPORTED;
      reason = "The measured profile did not pass deterministic output validation";
    } else if (platformMismatch) {
      status = OptimizationStatus.UNSUPPORTED;
      reason = "The measured profile does not cover this platform or processor topology";
    } else if (!mismatches.isEmpty() || !missingArguments.isEmpty()) {
      status = OptimizationStatus.DISABLED;
      reason =
          profile.javaLaunch().isPresent()
              ? "The measured profile requires a JVM restart with the recommended runtime options"
              : "The measured profile does not match the running JVM";
    } else {
      status = OptimizationStatus.ENABLED;
      reason = "The running model and JVM satisfy the exact measured ModelJars profile";
    }

    Map<String, String> settings = new LinkedHashMap<>();
    settings.put("profile-id", profile.id());
    settings.put("marker-coordinate", profile.markerCoordinate().toString());
    settings.put("recommendations", mapString(profile.recommendations()));
    profile
        .javaLaunch()
        .ifPresent(
            launch -> {
              settings.put("recommended-runtime", launch.runtime());
              settings.put("recommended-java-feature", Integer.toString(launch.javaFeature()));
              settings.put("required-jvm-arguments", String.join("\n", launch.jvmArguments()));
            });
    settings.put("missing-jvm-arguments", String.join("\n", missingArguments));
    settings.put("selector-mismatches", String.join("; ", mismatches));
    settings.put("benchmark-id", profile.evidence().benchmarkId());
    OptimizationDecision decision =
        new OptimizationDecision(decisionId(profile.id()), status, reason, settings);
    return new Evaluation(status == OptimizationStatus.ENABLED, decision);
  }

  private static void mergeRecommendations(
      Map<String, String> selected, ModelPerformanceProfile profile) {
    profile
        .recommendations()
        .forEach(
            (key, value) -> {
              String previous = selected.putIfAbsent(key, value);
              if (previous != null && !previous.equalsIgnoreCase(value)) {
                throw new ModelJarException(
                    "Conflicting performance recommendations for "
                        + key
                        + ": "
                        + previous
                        + " and "
                        + value);
              }
            });
  }

  private static List<String> selectorMismatches(
      Map<String, String> selector, Map<String, String> runtime) {
    return selector.entrySet().stream()
        .sorted(Map.Entry.comparingByKey())
        .filter(entry -> mismatch(entry.getValue(), runtime.get(entry.getKey())))
        .map(
            entry ->
                entry.getKey()
                    + " expected '"
                    + entry.getValue()
                    + "' but was '"
                    + runtime.getOrDefault(entry.getKey(), "<missing>")
                    + "'")
        .toList();
  }

  private static boolean mismatch(String expected, String actual) {
    return actual == null || !expected.equalsIgnoreCase(actual.trim());
  }

  private static String mapString(Map<String, String> values) {
    return values.entrySet().stream()
        .sorted(Map.Entry.comparingByKey())
        .map(entry -> entry.getKey() + "=" + entry.getValue())
        .collect(java.util.stream.Collectors.joining("\n"));
  }

  private static String decisionId(String profileId) {
    return "modeljars.profile." + profileId.replace('_', '.');
  }

  private record Evaluation(boolean applies, OptimizationDecision decision) {}
}
