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
package com.integrallis.models.bench.fusion;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class FrozenTuningTest {
  static final String RECORD =
      """
      {"schemaVersion":1,"kind":"frozen",
       "devSplit":{"idsSha256":"x","ids":["gsm8k-train-0001","gsm8k-train-0002"]},
       "weights":[{"members":["A","B","C"],"rule":"poe","weights":[0.2,0.3,0.5]}],
       "bestMember":{"name":"C"},
       "calibration":{"members":{"A":{"temperature":1.5},"B":{"temperature":0.9},"C":{"temperature":2.0}}}}
      """;

  @Test
  void refusesAMismatchedDigestAndResolvesTunedValues(@TempDir Path directory) throws Exception {
    Path file = directory.resolve("frozen.json");
    Files.writeString(file, RECORD, StandardCharsets.UTF_8);
    String sha = Digests.sha256(Files.readAllBytes(file));
    assertThatThrownBy(() -> FrozenTuning.load(file, "0".repeat(64)))
        .hasMessageContaining("digest mismatch");
    assertThatThrownBy(() -> FrozenTuning.load(file, null)).hasMessageContaining("--frozen-sha256");

    FrozenTuning frozen = FrozenTuning.load(file, sha);
    assertThat(ArmSpec.parse("fuse:A+B+C:poe:tuned").resolveWeights(frozen))
        .containsExactly(0.2, 0.3, 0.5);
    assertThatThrownBy(() -> ArmSpec.parse("fuse:A+B+C:mixture:tuned").resolveWeights(frozen))
        .hasMessageContaining("no tuned weights");
    assertThat(frozen.bestMember()).isEqualTo("C");
    assertThat(frozen.temperatures()).containsEntry("B", 0.9);
    assertThatThrownBy(
            () ->
                frozen.requireDisjoint(
                    List.of(new DatasetItem("gsm8k-train-0002", "q", "1", List.of()))))
        .hasMessageContaining("development split");
    frozen.requireDisjoint(List.of(new DatasetItem("gsm8k-test-0002", "q", "1", List.of())));
  }
}
