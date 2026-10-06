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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.bench.fusion.ReportModel.Item;
import java.time.Instant;
import java.util.List;
import java.util.Map;

/** Complete schema-v1 reports built from toy runs. */
final class ReportFixtures {
  private ReportFixtures() {}

  static ReportModel.Report armReport(ObjectMapper mapper, List<Item> items) {
    return armReport(
        mapper,
        items,
        "F-uniform",
        "fuse:A+B+C:poe:uniform",
        "fuse",
        List.of("A", "B", "C"),
        "sha-of-data");
  }

  static ReportModel.Report armReport(
      ObjectMapper mapper,
      List<Item> items,
      String label,
      String spec,
      String kind,
      List<String> members,
      String datasetSha) {
    List<ReportModel.MemberConfig> memberConfigs =
        members.stream()
            .map(
                name ->
                    new ReportModel.MemberConfig(
                        name,
                        "/models/" + name + ".gguf",
                        "a".repeat(64),
                        10,
                        "b".repeat(64),
                        Map.of(),
                        "c".repeat(64),
                        "qwen3",
                        7,
                        12,
                        null,
                        1))
            .toList();
    ReportModel.Config config =
        new ReportModel.Config(
            new ReportModel.ArmConfig(
                label,
                spec,
                spec,
                kind,
                members,
                "poe",
                List.of(1 / 3.0, 1 / 3.0, 1 / 3.0),
                "uniform",
                null,
                1,
                null,
                null,
                false,
                null,
                "number",
                List.of(),
                List.of()),
            new ReportModel.DecodingConfig(
                0f,
                0,
                1f,
                List.of(3L),
                24,
                "off",
                "chatml-no-think",
                "d".repeat(64),
                "resource",
                "e".repeat(64),
                3,
                2048,
                2,
                "generative",
                "sum",
                null,
                "argmax"),
            memberConfigs,
            new ReportModel.DatasetConfig(
                "gsm8k",
                "/data/gsm8k-test.jsonl",
                datasetSha,
                "openai/gsm8k",
                "rev",
                null,
                null,
                null,
                null,
                0,
                0,
                items.size(),
                DatasetItem.idsSha256(List.of())),
            new ReportModel.BackendConfig("pure-java", null, null),
            Map.of());
    return new ReportModel.Report(
        ReportModel.SCHEMA_VERSION,
        "arm",
        Instant.now().toString(),
        config,
        FusionEnvironment.capture("0".repeat(40), mapper),
        new ReportModel.Timings(1, 0.5, 1024, "test", 0),
        SummaryBuilder.summarize(items),
        items);
  }
}
