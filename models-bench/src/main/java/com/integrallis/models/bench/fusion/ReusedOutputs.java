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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.bench.fusion.ReportModel.MemberConfig;
import com.integrallis.models.bench.fusion.ReportModel.Output;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Member-arm reports reused by vote and rerank arms, so answer-level controls do not regenerate
 * (and do not re-pay) outputs that already exist under the same configuration.
 */
final class ReusedOutputs {

  record Entry(Output output, double coreSeconds, String renderedPromptSha256) {}

  record Loaded(
      Map<String, Map<String, Entry>> byMember,
      List<MemberConfig> members,
      List<String> paths,
      List<String> sha256s) {}

  private ReusedOutputs() {}

  static Loaded load(
      List<Path> reports,
      ObjectMapper mapper,
      String datasetSha256,
      boolean thinking,
      DecodeSettings settings,
      String promptsSha256)
      throws IOException {
    Map<String, Map<String, Entry>> byMember = new LinkedHashMap<>();
    List<MemberConfig> members = new ArrayList<>();
    List<String> paths = new ArrayList<>();
    List<String> shas = new ArrayList<>();
    for (Path path : reports) {
      byte[] bytes = Files.readAllBytes(path);
      JsonNode root = mapper.readTree(bytes);
      JsonNode config = root.path("config");
      require("arm".equals(root.path("kind").asText()), path, "is not an arm report");
      require(
          "member".equals(config.path("arm").path("kind").asText()),
          path,
          "is not a single-member arm");
      require(
          Objects.equals(datasetSha256, config.path("dataset").path("sha256").asText()),
          path,
          "was produced on a different dataset file");
      require(
          config.path("decoding").path("thinking").asText().equals(thinking ? "on" : "off"),
          path,
          "used a different thinking mode");
      require(
          config.path("decoding").path("maxTokens").asInt() == settings.maxTokens(),
          path,
          "used a different max-tokens cap");
      require(
          Float.compare(
                  (float) config.path("decoding").path("temperature").asDouble(),
                  settings.temperature())
              == 0,
          path,
          "used a different temperature");
      require(
          Objects.equals(promptsSha256, config.path("decoding").path("promptsSha256").asText()),
          path,
          "used a different prompt file");
      String member = config.path("arm").path("members").get(0).asText();
      require(!byMember.containsKey(member), path, "duplicates reused member " + member);
      Map<String, Entry> items = new LinkedHashMap<>();
      for (JsonNode item : root.path("items")) {
        Output original = mapper.treeToValue(item.path("outputs").get(0), Output.class);
        Output reused =
            new Output(
                original.producer(),
                original.seed(),
                original.text(),
                original.tokenIds(),
                original.tokenLogProbabilities(),
                original.reasoningAnswer(),
                original.statedAnswer(),
                original.confidence(),
                original.consistent(),
                original.thinkPresent(),
                original.thinkTruncated(),
                original.truncated(),
                original.stoppedOnEndOfGeneration(),
                original.tokens(),
                original.wallMillis(),
                original.prefillMillis(),
                original.memberForwardMillis(),
                null,
                List.of(),
                true);
        items.put(
            item.path("id").asText(),
            new Entry(
                reused,
                item.path("coreSeconds").asDouble(),
                item.path("renderedPromptSha256").asText()));
      }
      byMember.put(member, items);
      for (JsonNode memberConfig : config.path("members")) {
        members.add(mapper.treeToValue(memberConfig, MemberConfig.class));
      }
      paths.add(path.toAbsolutePath().toString());
      shas.add(Digests.sha256(bytes));
    }
    return new Loaded(byMember, members, paths, shas);
  }

  private static void require(boolean condition, Path path, String message) {
    if (!condition) {
      throw new IllegalArgumentException("reused report " + path + " " + message);
    }
  }
}
