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
package com.integrallis.models.rag;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.BackendDiagnostics;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The exercise report has to state the configuration it ran under.
 *
 * <p>A two-arm run that varied a native setting produced reports recording neither the property nor
 * any resolved backend diagnostic, so the treatment arm could not be shown to have differed from
 * its control. Four of five subjects changed verdict and the result had to be discarded, because
 * "the switch worked" and "the switch did nothing and the flaky case did not fire" leave an
 * identical artifact. An ablation whose toggle is absent from the evidence is not a measurement,
 * and these tests exist so that cannot recur silently.
 */
@Tag("unit")
class FrameworkExerciseReportSettingsTest {

  private static BackendDiagnostics diagnostics() {
    return new BackendDiagnostics(
        "rust-ffm", "plan-7", Map.of("native-quantized-decode", "true"), List.of());
  }

  @Test
  void recordsEveryModelsPropertyInForceSoAnArmCanBeToldFromItsControl() {
    Properties properties = new Properties();
    properties.setProperty("models.native.kernels.threads", "8");
    properties.setProperty("models.native.threadCount", "8");
    properties.setProperty("unrelated.property", "ignored");

    Map<String, Object> report = new LinkedHashMap<>();
    FrameworkExerciseCli.recordConfiguration(report, diagnostics(), properties);

    @SuppressWarnings("unchecked")
    Map<String, String> tuning = (Map<String, String>) report.get("modelsSystemProperties");
    assertThat(tuning)
        .containsEntry("models.native.kernels.threads", "8")
        .containsEntry("models.native.threadCount", "8");
    assertThat(tuning).doesNotContainKey("unrelated.property");
  }

  @Test
  void recordsAnEmptyTuningMapRatherThanOmittingIt() {
    // A library-defaults arm must be distinguishable from an arm whose settings were never
    // captured. Absent and empty are different claims, so the key is always present.
    Map<String, Object> report = new LinkedHashMap<>();
    FrameworkExerciseCli.recordConfiguration(report, diagnostics(), new Properties());

    assertThat(report).containsKey("modelsSystemProperties");
    @SuppressWarnings("unchecked")
    Map<String, String> tuning = (Map<String, String>) report.get("modelsSystemProperties");
    assertThat(tuning).isEmpty();
  }

  @Test
  void recordsWhatTheBackendResolvedToNotOnlyWhatWasRequested() {
    Map<String, Object> report = new LinkedHashMap<>();
    FrameworkExerciseCli.recordConfiguration(report, diagnostics(), new Properties());

    assertThat(report).containsEntry("backendPlanVersion", "plan-7");
    @SuppressWarnings("unchecked")
    Map<String, String> environment = (Map<String, String>) report.get("backendEnvironment");
    // This is the half that catches a silent fallback: the requested name was always recorded, the
    // resolved diagnostics were not.
    assertThat(environment).containsEntry("native-quantized-decode", "true");
  }

  @Test
  void ordersTheRecordedSettingsSoTwoArmsDiffByContentRatherThanByIterationOrder() {
    Properties first = new Properties();
    first.setProperty("models.b", "2");
    first.setProperty("models.a", "1");
    Properties second = new Properties();
    second.setProperty("models.a", "1");
    second.setProperty("models.b", "2");

    Map<String, Object> firstReport = new LinkedHashMap<>();
    Map<String, Object> secondReport = new LinkedHashMap<>();
    FrameworkExerciseCli.recordConfiguration(firstReport, diagnostics(), first);
    FrameworkExerciseCli.recordConfiguration(secondReport, diagnostics(), second);

    // Two arms are compared by diffing their artifacts, so an unstable key order would show as a
    // difference that is not one.
    assertThat(firstReport.get("modelsSystemProperties").toString())
        .isEqualTo(secondReport.get("modelsSystemProperties").toString());
  }
}
