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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ActivatedHiddenDecisionProfileCliTest {

  @Test
  void loadsOnlyTheFrozenValidatedHead(@TempDir Path directory) throws Exception {
    var mapper = ActivatedDecisionProfileCli.mapper();
    ObjectNode artifact = mapper.createObjectNode();
    artifact.put("schemaVersion", 1);
    artifact.put("experiment", ActivatedHiddenDecisionProfileCli.EXPERIMENT);
    artifact.put("artifactSha256", ActivatedHiddenDecisionProfileCli.HEAD_ARTIFACT_SHA256);
    artifact.put("adapterSha256", ActivatedHiddenDecisionProfileCli.ADAPTER_SHA256);
    artifact.put("validationPassed", true);
    artifact
        .putObject("base")
        .put("productionGgufSha256", ActivatedHiddenDecisionProfileCli.MODEL_SHA256);
    artifact.putObject("representation").put("hiddenDimension", 2048);
    ObjectNode normalization = artifact.putObject("normalization");
    ArrayNode mean = normalization.putArray("mean");
    ArrayNode scale = normalization.putArray("scale");
    ArrayNode weight = artifact.putObject("classifier").putArray("weight");
    for (int index = 0; index < 2048; index++) {
      mean.add(0);
      scale.add(1);
      weight.add(index == 0 ? 1 : 0);
    }
    artifact.withObject("classifier").put("bias", -0.5);
    Path path = directory.resolve("head.json");
    mapper.writeValue(path.toFile(), artifact);

    var head = ActivatedHiddenDecisionProfileCli.loadHead(mapper, path);

    float[] hidden = new float[2048];
    hidden[0] = 1;
    assertThat(head.score(hidden).score()).isEqualTo(0.5);
  }

  @Test
  void scoresTheTwoClassesAtTheFixedZeroBoundary() {
    var observations =
        List.of(
            observation(true, 1.0),
            observation(true, 0.0),
            observation(false, 0.0),
            observation(false, 1.0));

    var score = ActivatedHiddenDecisionProfileCli.score(observations);

    assertThat(score.correctCalls()).isEqualTo(1);
    assertThat(score.correctNoCalls()).isEqualTo(1);
    assertThat(score.balancedAccuracy()).isEqualTo(0.5);
  }

  private static ActivatedHiddenDecisionProfileCli.Observation observation(
      boolean expected, double score) {
    return new ActivatedHiddenDecisionProfileCli.Observation(
        "id", "kind", expected, 1, 1, 1, true, score, score > 0, 1);
  }
}
