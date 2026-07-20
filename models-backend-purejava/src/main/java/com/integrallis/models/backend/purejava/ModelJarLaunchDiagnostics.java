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

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.modeljars.JavaLaunchProfile;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelPerformanceProfile;
import org.modeljars.ModelPerformanceProfileRegistry;

/** Adds exact ModelJars startup-profile compliance to pure-Java backend diagnostics. */
final class ModelJarLaunchDiagnostics {

  private static final Set<String> RESTART_SELECTOR_KEYS =
      Set.of("compiler", "java-feature", "java-version", "vm-name", "vm-vendor", "vm-version");

  private ModelJarLaunchDiagnostics() {}

  static BackendDiagnostics enrich(
      BackendDiagnostics base,
      ModelJarDescriptor descriptor,
      ModelPerformanceProfileRegistry registry,
      Map<String, String> runtime,
      List<String> inputArguments) {
    Objects.requireNonNull(base, "base");
    Objects.requireNonNull(descriptor, "descriptor");
    Objects.requireNonNull(registry, "registry");
    Objects.requireNonNull(runtime, "runtime");
    Objects.requireNonNull(inputArguments, "inputArguments");

    Map<String, String> environment = new LinkedHashMap<>(base.environment());
    environment.put("modeljar-alias", descriptor.alias());
    environment.put("modeljar-coordinate", descriptor.markerCoordinate().toString());
    descriptor.sha256().ifPresent(value -> environment.put("modeljar-sha256", value));

    List<OptimizationDecision> decisions = new ArrayList<>(base.optimizations());
    registry.profilesFor(descriptor).stream()
        .filter(profile -> "pure-java".equals(profile.backend()))
        .filter(profile -> profile.javaLaunch().isPresent())
        .map(profile -> decision(profile, runtime, inputArguments))
        .forEach(decisions::add);
    return new BackendDiagnostics(base.backend(), base.planVersion(), environment, decisions);
  }

  private static OptimizationDecision decision(
      ModelPerformanceProfile profile, Map<String, String> runtime, List<String> inputArguments) {
    JavaLaunchProfile launch = profile.javaLaunch().orElseThrow();
    List<String> mismatches = selectorMismatches(profile.runtimeSelector(), runtime);
    boolean platformMismatch =
        profile.runtimeSelector().keySet().stream()
            .filter(key -> !RESTART_SELECTOR_KEYS.contains(key))
            .anyMatch(key -> mismatch(profile.runtimeSelector().get(key), runtime.get(key)));
    List<String> missingArguments = launch.missingArguments(inputArguments);

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
      reason = "The measured profile requires a JVM restart with the recommended runtime options";
    } else {
      status = OptimizationStatus.ENABLED;
      reason = "The running JVM satisfies the exact measured ModelJars launch profile";
    }

    Map<String, String> settings = new LinkedHashMap<>();
    settings.put("profile-id", profile.id());
    settings.put("marker-coordinate", profile.markerCoordinate().toString());
    settings.put("recommended-runtime", launch.runtime());
    settings.put("recommended-java-feature", Integer.toString(launch.javaFeature()));
    settings.put("required-jvm-arguments", String.join("\n", launch.jvmArguments()));
    settings.put("missing-jvm-arguments", String.join("\n", missingArguments));
    settings.put("selector-mismatches", String.join("; ", mismatches));
    settings.put("benchmark-id", profile.evidence().benchmarkId());
    return new OptimizationDecision(decisionId(profile.id()), status, reason, settings);
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

  private static String decisionId(String profileId) {
    return "modeljars.profile." + profileId.replace('_', '.');
  }
}
