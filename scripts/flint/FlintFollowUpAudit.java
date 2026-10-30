/*
 * Copyright 2025-2026 Integrallis Software, LLC
 * SPDX-License-Identifier: Apache-2.0
 */
package com.integrallis.models.bench;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;

/** Reassesses saved output only; never loads a model or changes the original report. */
public final class FlintFollowUpAudit {
  private FlintFollowUpAudit() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 3 || !args[2].matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("usage: FlintFollowUpAudit REPORT OUTPUT SCORER_REVISION");
    }
    Path source = Path.of(args[0]);
    Path output = Path.of(args[1]);
    if (source.toAbsolutePath().normalize().equals(output.toAbsolutePath().normalize())
        || (Files.exists(output) && Files.isSameFile(source, output))) {
      throw new IllegalArgumentException("The original report must remain unchanged");
    }
    var mapper = new ObjectMapper();
    var report = mapper.readTree(source.toFile());
    var candidate = ToolCallingCandidate.FLINT_40B_A2_5B;
    if (!report.path("complete").asBoolean()
        || !report.path("modelId").asText().equals(candidate.modelId())
        || !report.path("artifactSha256").asText().equals(candidate.artifactSha256())
        || !report.path("followUp").path("applicable").asBoolean()) {
      throw new IllegalArgumentException("Expected a complete pinned Flint tool report with follow-up");
    }
    var weather =
        Needle2ToolQualification.loadSuite(mapper).cases().stream()
            .filter(item -> item.id().equals("spring-weather-zipcode"))
            .findFirst()
            .orElseThrow();
    var original = report.path("followUp");
    var assessed =
        ToolCallingQualificationCli.evaluateFollowUp(
            mapper,
            candidate.template().toolSyntax(),
            weather.toolSpecs(mapper),
            original.path("output").asText(),
            original.path("endToEndMillis").asLong());
    var audit = mapper.createObjectNode();
    audit.put("kind", "post-hoc-tool-follow-up-assessment");
    audit.put("newInference", false);
    audit.put("sourceReportSha256", Hashing.sha256(source));
    audit.put("runtimeSourceRevision", report.path("modelsRevision").asText());
    audit.put("scorerSourceRevision", args[2]);
    audit.put("originalPolicyVersion", report.path("policyVersion").asText());
    audit.put("policyVersion", ToolCallingQualificationCli.POLICY_VERSION);
    audit.put("originalFollowUpPassed", original.path("passed").asBoolean());
    audit.set("followUp", mapper.valueToTree(assessed));
    Files.createDirectories(output.toAbsolutePath().getParent());
    mapper.writerWithDefaultPrettyPrinter().writeValue(output.toFile(), audit);
  }
}
