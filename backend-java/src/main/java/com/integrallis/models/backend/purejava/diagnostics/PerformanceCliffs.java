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
package com.integrallis.models.backend.purejava.diagnostics;

import com.integrallis.models.api.BackendDiagnostics;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.StringJoiner;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * Process-wide, once-per-reason registry of {@link PerformanceCliff} reports.
 *
 * <p>Code that takes a slower path calls {@link #report(PerformanceCliff, String)} from the branch
 * itself. The first report of each reason in a process commits one {@link PerformanceCliffEvent}
 * and records its detail; every later report of that reason is a single volatile read and returns
 * {@code false}, so a report inside a per-token loop costs nothing after the first token.
 *
 * <p>Reports are surfaced in backend diagnostics as {@value #ENVIRONMENT_KEY} (comma-separated ids)
 * and {@value #ENVIRONMENT_KEY_PREFIX}{@code <id>} (the first detail). They are process scoped: a
 * cliff taken while serving one model is visible in the diagnostics of every backend in the same
 * JVM, because the fallback (for example a vector width cap) usually is too.
 */
public final class PerformanceCliffs {

  /** Diagnostics environment key listing every reported cliff id, comma separated. */
  public static final String ENVIRONMENT_KEY = "performance-cliffs";

  /** Prefix of the per-cliff diagnostics environment key; the value is the first detail. */
  public static final String ENVIRONMENT_KEY_PREFIX = "performance-cliff.";

  private static final PerformanceCliff[] CLIFFS = PerformanceCliff.values();
  private static final AtomicReferenceArray<String> DETAILS =
      new AtomicReferenceArray<>(CLIFFS.length);

  private PerformanceCliffs() {}

  /**
   * Reports that the fast path named by {@code cliff} was not taken.
   *
   * @param cliff the reason
   * @param detail branch context (architecture, tensor types, kernel, widths); {@code null} is
   *     recorded as an empty string
   * @return {@code true} only for the first report of {@code cliff} in this process (since the last
   *     {@link #reset()}), which is the report that committed the JFR event
   */
  public static boolean report(PerformanceCliff cliff, String detail) {
    int index = cliff.ordinal();
    if (DETAILS.get(index) != null) {
      return false;
    }
    String recorded = detail == null ? "" : detail;
    if (!DETAILS.compareAndSet(index, null, recorded)) {
      return false;
    }
    PerformanceCliffEvent event = new PerformanceCliffEvent(cliff.id(), recorded);
    if (event.shouldCommit()) {
      event.commit();
    }
    return true;
  }

  /** Returns every cliff reported so far with its first detail, in declaration order. */
  public static Map<PerformanceCliff, String> reported() {
    Map<PerformanceCliff, String> reported = new EnumMap<>(PerformanceCliff.class);
    for (PerformanceCliff cliff : CLIFFS) {
      String detail = DETAILS.get(cliff.ordinal());
      if (detail != null) {
        reported.put(cliff, detail);
      }
    }
    return reported;
  }

  /**
   * Returns the diagnostics environment entries for the cliffs reported so far, or an empty map
   * when none has been reported.
   */
  public static Map<String, String> environment() {
    Map<PerformanceCliff, String> reported = reported();
    if (reported.isEmpty()) {
      return Map.of();
    }
    Map<String, String> environment = new LinkedHashMap<>();
    StringJoiner ids = new StringJoiner(",");
    reported.forEach(
        (cliff, detail) -> {
          ids.add(cliff.id());
          environment.put(ENVIRONMENT_KEY_PREFIX + cliff.id(), detail);
        });
    environment.put(ENVIRONMENT_KEY, ids.toString());
    return environment;
  }

  /**
   * Returns {@code diagnostics} with the current {@link #environment()} entries added. When no
   * cliff has been reported the argument itself is returned, so diagnostics on the fast path are
   * unchanged.
   */
  public static BackendDiagnostics enrich(BackendDiagnostics diagnostics) {
    Objects.requireNonNull(diagnostics, "diagnostics");
    Map<String, String> cliffs = environment();
    if (cliffs.isEmpty()) {
      return diagnostics;
    }
    Map<String, String> environment = new LinkedHashMap<>(diagnostics.environment());
    environment.putAll(cliffs);
    return new BackendDiagnostics(
        diagnostics.backend(), diagnostics.planVersion(), environment, diagnostics.optimizations());
  }

  /**
   * Forgets every report so the next report of each reason commits a new event. Intended for tests
   * and for hosts that want a fresh report after reconfiguring and reloading models.
   */
  public static void reset() {
    for (int index = 0; index < CLIFFS.length; index++) {
      DETAILS.set(index, null);
    }
  }
}
