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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;

/** Applies the production policy to file-backed reports and emits hashed evidence. */
public final class RagQualificationCli {
  private static final ObjectMapper MAPPER =
      new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

  private RagQualificationCli() {}

  public static void main(String[] args) throws Exception {
    run(args);
  }

  static RagQualificationEvidence run(String[] args) throws IOException {
    Arguments arguments = Arguments.parse(args);
    RagBenchmarkReport candidate = readReport(arguments.candidate());
    List<RagBenchmarkReport> comparators =
        arguments.comparators().stream().map(RagQualificationCli::readReportUnchecked).toList();
    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, comparators);
    RagQualificationEvidence evidence =
        new RagQualificationEvidence(
            RagQualificationEvidence.CURRENT_SCHEMA_VERSION,
            RagProductionQualificationPolicy.POLICY_ID,
            reference(arguments.candidate()),
            arguments.comparators().stream().map(RagQualificationCli::referenceUnchecked).toList(),
            qualification);

    if (arguments.output() == null) {
      MAPPER.writeValue(System.out, evidence);
      System.out.println();
    } else {
      Path parent = arguments.output().toAbsolutePath().getParent();
      if (parent != null) {
        Files.createDirectories(parent);
      }
      MAPPER.writeValue(arguments.output().toFile(), evidence);
    }

    if (arguments.requireQualified() && !qualification.qualified()) {
      throw new IllegalStateException("model qualification failed: " + qualification.verdict());
    }
    return evidence;
  }

  private static RagBenchmarkReport readReport(Path path) throws IOException {
    if (!Files.isRegularFile(path)) {
      throw new IllegalArgumentException("benchmark report does not exist: " + path);
    }
    RagBenchmarkReport report = MAPPER.readValue(path.toFile(), RagBenchmarkReport.class);
    if (report.schemaVersion() != RagBenchmarkReport.CURRENT_SCHEMA_VERSION) {
      throw new IllegalArgumentException(
          "unsupported benchmark report schema " + report.schemaVersion() + " in " + path);
    }
    return report;
  }

  private static RagBenchmarkReport readReportUnchecked(Path path) {
    try {
      return readReport(path);
    } catch (IOException failure) {
      throw new IllegalArgumentException("cannot read benchmark report: " + path, failure);
    }
  }

  private static RagQualificationReportReference reference(Path path) throws IOException {
    try {
      byte[] digest = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path));
      return new RagQualificationReportReference(path.toString(), HexFormat.of().formatHex(digest));
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException(impossible);
    }
  }

  private static RagQualificationReportReference referenceUnchecked(Path path) {
    try {
      return reference(path);
    } catch (IOException failure) {
      throw new IllegalArgumentException("cannot hash benchmark report: " + path, failure);
    }
  }

  private record Arguments(
      Path candidate, List<Path> comparators, Path output, boolean requireQualified) {

    private Arguments {
      comparators = List.copyOf(comparators);
    }

    private static Arguments parse(String[] args) {
      Path candidate = null;
      List<Path> comparators = new ArrayList<>();
      Path output = null;
      boolean requireQualified = false;

      for (int index = 0; index < args.length; index++) {
        switch (args[index]) {
          case "--candidate" -> {
            candidate = Path.of(requireValue(args, ++index, "--candidate"));
          }
          case "--comparator" -> {
            comparators.add(Path.of(requireValue(args, ++index, "--comparator")));
          }
          case "--output" -> {
            output = Path.of(requireValue(args, ++index, "--output"));
          }
          case "--require-qualified" -> requireQualified = true;
          default ->
              throw new IllegalArgumentException("unknown qualification option: " + args[index]);
        }
      }
      if (candidate == null) {
        throw new IllegalArgumentException("--candidate is required");
      }
      if (comparators.isEmpty()) {
        throw new IllegalArgumentException("at least one --comparator is required");
      }
      return new Arguments(candidate, comparators, output, requireQualified);
    }

    private static String requireValue(String[] args, int index, String option) {
      if (index >= args.length || args[index].startsWith("--")) {
        throw new IllegalArgumentException(option + " requires a value");
      }
      return args[index];
    }
  }
}
