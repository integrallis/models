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

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/** Validated native settings selected from deployment overrides and exact ModelJar profiles. */
record NativeKernelSettings(boolean nativeDecode, boolean q5_0Grouped, int threadCount) {
  private static final String PROPERTY_PREFIX = "models.native.";
  private static final Set<String> SUPPORTED_SETTINGS =
      Set.of(
          RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY,
          RustGgufBatchedMatrixKernel.Q5_0_GROUPED_PROPERTY,
          NativeKernelLibrary.THREAD_COUNT_PROPERTY);

  NativeKernelSettings {
    if (threadCount < 1 || threadCount > 256) {
      throw new IllegalArgumentException("threadCount must be between 1 and 256: " + threadCount);
    }
  }

  static NativeKernelSettings fromSystemProperties(Map<String, String> recommendations) {
    Properties systemProperties = System.getProperties();
    Map<String, String> deployment = new LinkedHashMap<>();
    synchronized (systemProperties) {
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
        recommendations, Map.copyOf(deployment), Runtime.getRuntime().availableProcessors());
  }

  static NativeKernelSettings resolve(
      Map<String, String> recommendations, Map<String, String> deployment, int defaultThreadCount) {
    Objects.requireNonNull(recommendations, "recommendations");
    Objects.requireNonNull(deployment, "deployment");
    validateSettings(recommendations, "recommendation");
    validateSettings(deployment, "deployment setting");
    return new NativeKernelSettings(
        booleanSetting(
            RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY,
            configured(
                RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY, deployment, recommendations)),
        booleanSetting(
            RustGgufBatchedMatrixKernel.Q5_0_GROUPED_PROPERTY,
            configured(
                RustGgufBatchedMatrixKernel.Q5_0_GROUPED_PROPERTY, deployment, recommendations)),
        threadCount(
            configured(NativeKernelLibrary.THREAD_COUNT_PROPERTY, deployment, recommendations),
            defaultThreadCount));
  }

  private static String configured(
      String property, Map<String, String> deployment, Map<String, String> recommendations) {
    String value = deployment.get(property);
    return value != null ? value : recommendations.get(property);
  }

  private static boolean booleanSetting(String property, String value) {
    if (value == null) {
      return false;
    }
    if (value.equalsIgnoreCase("true")) {
      return true;
    }
    if (value.equalsIgnoreCase("false")) {
      return false;
    }
    throw new IllegalArgumentException(property + " must be true or false: " + value);
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
        .findFirst()
        .ifPresent(
            key -> {
              throw new IllegalArgumentException("Unsupported native " + source + ": " + key);
            });
  }
}
