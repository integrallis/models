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
package com.integrallis.models.decisions;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

/** A run has to be able to say what it ran on, and a caller has to be able to insist. */
final class RuntimeReportTest {

  @Test
  void carriesTheResolvedConfigurationInOrder() {
    RuntimeReport report =
        new RuntimeReport()
            .with("model", "qwen3-0.6b")
            .kernel("cuda", "NVIDIA A40", "cc 86, ptx sm_80");

    assertThat(report.toString())
        .isEqualTo(
            "runtime[model=qwen3-0.6b, kernel=cuda, device=NVIDIA A40, "
                + "kernelDetail=cc 86, ptx sm_80]");
    assertThat(report.get("kernel")).isEqualTo("cuda");
  }

  @Test
  void omitsADeviceWhenThereIsNone() {
    RuntimeReport report = new RuntimeReport().kernel("rust-ffm", "", "");
    assertThat(report.toString()).isEqualTo("runtime[kernel=rust-ffm]");
    assertThat(report.get("device")).isEmpty();
  }

  @Test
  void requireAcceptsAMatch() {
    new RuntimeReport().kernel("cuda", "NVIDIA A40", "").require("kernel", "cuda");
  }

  @Test
  void requireRejectsAFallback() {
    // The case this type exists for: a GPU comparison that quietly ran on the CPU is not a
    // slower result, it is a different experiment, and it must not be reportable as the first.
    assertThatThrownBy(
            () -> new RuntimeReport().kernel("rust-ffm", "", "").require("kernel", "cuda"))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("expected kernel=cuda")
        .hasMessageContaining("resolved kernel=rust-ffm");
  }

  @Test
  void requireRejectsAnUnsetFactRatherThanPassingVacuously() {
    assertThatThrownBy(() -> new RuntimeReport().require("kernel", "cuda"))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("<unset>");
  }

  @Test
  void emitsJsonForAnEvidenceFile() {
    String json =
        new RuntimeReport().with("model", "qwen3-0.6b").kernel("cuda", "A40", "").toJson();
    assertThat(json).isEqualTo("{\"model\":\"qwen3-0.6b\",\"kernel\":\"cuda\",\"device\":\"A40\"}");
  }

  @Test
  void escapesQuotesSoAnAwkwardDeviceNameCannotBreakTheFile() {
    assertThat(new RuntimeReport().with("device", "a \"quoted\" name").toJson())
        .isEqualTo("{\"device\":\"a \\\"quoted\\\" name\"}");
  }
}
