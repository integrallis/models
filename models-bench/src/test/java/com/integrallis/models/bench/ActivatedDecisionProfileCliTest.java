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

import java.util.List;
import org.junit.jupiter.api.Test;

class ActivatedDecisionProfileCliTest {

  @Test
  void calibratesOnlyOnTheDeclaredCalibrationPartition() {
    List<ActivatedDecisionProfileCli.Observation> observations =
        List.of(
            observation("c1", true, true, 3),
            observation("c2", true, true, 2),
            observation("n1", true, false, -1),
            observation("n2", true, false, -2),
            observation("screen-call", false, true, -100),
            observation("screen-none", false, false, 100));

    ActivatedDecisionProfileCli.Calibration result =
        ActivatedDecisionProfileCli.calibrate(observations);

    assertThat(result.threshold()).isEqualTo(0.5f);
    assertThat(result.calibration().balancedAccuracy()).isEqualTo(1.0);
    assertThat(result.calibration().correctCalls()).isEqualTo(2);
    assertThat(result.calibration().correctNoCalls()).isEqualTo(2);
    assertThat(result.screen().correctCalls()).isZero();
    assertThat(result.screen().correctNoCalls()).isZero();
  }

  @Test
  void resolvesAnAccuracyTieTowardFewerFalseCalls() {
    List<ActivatedDecisionProfileCli.Observation> observations =
        List.of(
            observation("c1", true, true, 1),
            observation("c2", true, true, 1),
            observation("n1", true, false, 1),
            observation("n2", true, false, 1),
            observation("screen-call", false, true, 1),
            observation("screen-none", false, false, 1));

    ActivatedDecisionProfileCli.Calibration result =
        ActivatedDecisionProfileCli.calibrate(observations);

    assertThat(result.threshold()).isEqualTo(1.0f);
    assertThat(result.calibration().correctCalls()).isZero();
    assertThat(result.calibration().correctNoCalls()).isEqualTo(2);
  }

  @Test
  void preservesTheFrozenSchemaMemberOrderWhenReconstructingPrompts() throws Exception {
    var mapper = ActivatedDecisionProfileCli.mapper();
    var schema = mapper.readTree("{\"type\":\"object\",\"properties\":{\"z\":{},\"a\":{}}}");
    String serialized = mapper.writeValueAsString(schema);

    assertThat(serialized.indexOf("\"type\"")).isLessThan(serialized.indexOf("\"properties\""));
    assertThat(serialized.indexOf("\"z\"")).isLessThan(serialized.indexOf("\"a\""));
  }

  private static ActivatedDecisionProfileCli.Observation observation(
      String id, boolean calibration, boolean callExpected, float margin) {
    return new ActivatedDecisionProfileCli.Observation(
        id,
        callExpected ? "simple" : "irrelevance",
        calibration ? "calibration" : "screen",
        callExpected,
        100,
        80,
        true,
        1,
        1,
        margin,
        1);
  }
}
