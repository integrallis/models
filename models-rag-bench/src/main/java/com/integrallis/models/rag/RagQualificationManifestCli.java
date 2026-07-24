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
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/** Generates and verifies a production qualification manifest from raw RAG reports. */
public final class RagQualificationManifestCli {
  private static final Set<String> OPTIONS =
      Set.of("reports", "output", "models-revision", "target");

  private RagQualificationManifestCli() {}

  public static void main(String[] args) throws Exception {
    run(args);
  }

  static void run(String[] args) throws IOException {
    Map<String, String> options = parse(args);
    Path reports = Path.of(required(options, "reports"));
    Path output = Path.of(required(options, "output"));
    String revision = required(options, "models-revision");
    int target = positiveInteger(required(options, "target"), "target");

    RagQualificationManifest manifest =
        RagQualificationManifestGenerator.generate(reports, revision, target);
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    new ObjectMapper()
        .enable(SerializationFeature.INDENT_OUTPUT)
        .writeValue(output.toFile(), manifest);
    manifest.requireTarget();
    System.out.printf(
        "Qualification target met: %d/%d models; manifest: %s%n",
        manifest.qualifiedModels(), manifest.targetQualifiedModels(), output.toAbsolutePath());
  }

  private static Map<String, String> parse(String[] args) {
    if (args.length % 2 != 0) {
      throw new IllegalArgumentException("every option requires a value");
    }
    Map<String, String> options = new HashMap<>();
    for (int index = 0; index < args.length; index += 2) {
      String raw = args[index];
      if (!raw.startsWith("--")) {
        throw new IllegalArgumentException("expected option, got: " + raw);
      }
      String name = raw.substring(2);
      if (!OPTIONS.contains(name)) {
        throw new IllegalArgumentException("unknown option: " + raw);
      }
      options.put(name, args[index + 1]);
    }
    return options;
  }

  private static String required(Map<String, String> options, String name) {
    String value = options.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }

  private static int positiveInteger(String raw, String name) {
    try {
      int value = Integer.parseInt(raw);
      if (value < 1) {
        throw new IllegalArgumentException("--" + name + " must be positive");
      }
      return value;
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException("--" + name + " must be an integer", failure);
    }
  }
}
