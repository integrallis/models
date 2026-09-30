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
import java.util.ArrayList;
import java.util.List;

/**
 * Required fields of a schemaVersion 1 report. A key that is absent is an error; a key that must
 * carry a value is an error when null. Every command validates before writing.
 */
final class ReportSchema {

  private ReportSchema() {}

  static final List<String> ALWAYS_PRESENT =
      List.of(
          "/schemaVersion!",
          "/kind!",
          "/createdAt!",
          "/config",
          "/environment/modelsCommit",
          "/environment/modelsRevisionArgument",
          "/environment/modelsDirty",
          "/environment/javaVersionOutput!",
          "/environment/jvmInputArguments!",
          "/environment/inferenceSystemProperties!",
          "/environment/vectorRuntime!",
          "/environment/availableProcessors!",
          "/environment/host/hostname!",
          "/environment/host/cpuModel!",
          "/environment/host/cpuFlags!",
          "/environment/host/logicalCores!",
          "/environment/host/ramBytes!",
          "/environment/host/osName!",
          "/environment/host/osVersion!",
          "/environment/host/arch!",
          "/timings/wallMillis!",
          "/timings/processCpuSeconds!",
          "/timings/peakRssBytes!",
          "/timings/peakRssMethod!",
          "/summary",
          "/items");

  static final List<String> ARM =
      List.of(
          "/environment/modelsRevisionArgument!",
          "/config/arm/label",
          "/config/arm/spec!",
          "/config/arm/canonical!",
          "/config/arm/kind!",
          "/config/arm/members!",
          "/config/arm/rule",
          "/config/arm/weights",
          "/config/arm/weightSource",
          "/config/arm/tieBreakMember",
          "/config/arm/samples!",
          "/config/arm/frozenPath",
          "/config/arm/frozenSha256",
          "/config/arm/pilot!",
          "/config/arm/pilotSubstitution",
          "/config/arm/scorer!",
          "/config/decoding/temperature!",
          "/config/decoding/topK!",
          "/config/decoding/topP!",
          "/config/decoding/seeds!",
          "/config/decoding/maxTokens!",
          "/config/decoding/thinking!",
          "/config/decoding/chatTemplate!",
          "/config/decoding/chatTemplateSha256!",
          "/config/decoding/promptsSha256!",
          "/config/decoding/memberThreads!",
          "/config/decoding/contextLength!",
          "/config/decoding/tokenLogEvery!",
          "/config/decoding/arcScoring!",
          "/config/decoding/rerankScore!",
          "/config/dataset/name!",
          "/config/dataset/path!",
          "/config/dataset/sha256!",
          "/config/dataset/hfRepo",
          "/config/dataset/hfRevision",
          "/config/dataset/devIdsSha256",
          "/config/dataset/offset!",
          "/config/dataset/limit!",
          "/config/dataset/itemCount!",
          "/config/dataset/selectedIdsSha256!",
          "/config/backend/name!",
          "/config/backend/nativeLibraryPath",
          "/config/backend/nativeLibrarySha256",
          "/summary/items!",
          "/summary/correct!",
          "/summary/accuracy!",
          "/summary/truncationRate!",
          "/summary/extractionFailures!",
          "/summary/generatedTokens!",
          "/summary/tokensPerSecond!",
          "/summary/coreSeconds!",
          "/summary/p95TraceTokens!");

  static final List<String> MEMBER =
      List.of(
          "name!",
          "path!",
          "sha256!",
          "sizeBytes!",
          "tokenizerSha256!",
          "chatTemplateSha256",
          "architecture!",
          "diagnostics");

  static final List<String> ITEM =
      List.of(
          "id!",
          "gold!",
          "renderedPromptSha256!",
          "prediction",
          "correct!",
          "truncated!",
          "outputs!",
          "coreSeconds!",
          "wallMillis!");

  static final List<String> OUTPUT =
      List.of(
          "producer!",
          "text!",
          "tokenIds!",
          "reasoningAnswer",
          "statedAnswer",
          "confidence",
          "consistent!",
          "truncated!",
          "memberForwardMillis!");

  /** Returns every violated requirement; empty when the report is valid. */
  static List<String> violations(JsonNode report) {
    List<String> problems = new ArrayList<>();
    check(report, ALWAYS_PRESENT, "", problems);
    String kind = report.path("kind").asText();
    boolean loadsModels =
        !report.path("config").path("members").isMissingNode()
            && report.path("config").path("members").isArray()
            && !report.path("config").path("members").isEmpty();
    if (loadsModels) {
      JsonNode members = report.path("config").path("members");
      for (int i = 0; i < members.size(); i++) {
        check(members.get(i), MEMBER, "/config/members/" + i + "/", problems);
      }
    }
    if ("arm".equals(kind)) {
      check(report, ARM, "", problems);
      if (!loadsModels
          && !report.path("config").path("arm").path("reusedMemberReports").isArray()) {
        problems.add("/config/members must list the members of an arm");
      }
      JsonNode items = report.path("items");
      for (int i = 0; i < items.size(); i++) {
        check(items.get(i), ITEM, "/items/" + i + "/", problems);
        JsonNode outputs = items.get(i).path("outputs");
        for (int j = 0; j < outputs.size(); j++) {
          check(outputs.get(j), OUTPUT, "/items/" + i + "/outputs/" + j + "/", problems);
        }
      }
    }
    return problems;
  }

  private static void check(
      JsonNode node, List<String> requirements, String prefix, List<String> problems) {
    for (String requirement : requirements) {
      boolean nonNull = requirement.endsWith("!");
      String path = nonNull ? requirement.substring(0, requirement.length() - 1) : requirement;
      String pointer = path.startsWith("/") ? path : "/" + path;
      JsonNode value = node.at(pointer);
      if (value.isMissingNode()) {
        problems.add(prefix + trim(path) + " is missing");
      } else if (nonNull && value.isNull()) {
        problems.add(prefix + trim(path) + " must not be null");
      }
    }
  }

  private static String trim(String path) {
    return path.startsWith("/") && !path.isEmpty() ? path.substring(1) : path;
  }
}
