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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Reads backend reports, validates comparability, and writes JSON plus Markdown evidence. */
final class BenchmarkComparisonCli {

  private static final Set<String> OPTIONS =
      Set.of("pure-java", "llama-cpp", "ollama", "json", "markdown");

  private BenchmarkComparisonCli() {}

  static void run(String[] args) throws IOException {
    Map<String, String> options = parse(args);
    ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    List<BenchmarkReport> reports = new ArrayList<>();
    reports.add(read(mapper, required(options, "pure-java")));
    reports.add(read(mapper, required(options, "llama-cpp")));
    if (options.containsKey("ollama")) {
      reports.add(read(mapper, options.get("ollama")));
    }
    ComparisonReport comparison = BenchmarkComparison.compare(reports);
    Path json = Path.of(required(options, "json"));
    Path markdown = Path.of(required(options, "markdown"));
    createParent(json);
    createParent(markdown);
    mapper.writeValue(json.toFile(), comparison);
    Files.writeString(markdown, ComparisonMarkdown.render(comparison));
  }

  private static BenchmarkReport read(ObjectMapper mapper, String path) throws IOException {
    return mapper.readValue(Path.of(path).toFile(), BenchmarkReport.class);
  }

  private static void createParent(Path output) throws IOException {
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
  }

  private static Map<String, String> parse(String[] args) {
    Map<String, String> values = new HashMap<>();
    for (int index = 0; index < args.length; index += 2) {
      if (!args[index].startsWith("--") || index + 1 >= args.length) {
        throw new IllegalArgumentException("comparison options must use --name value pairs");
      }
      String name = args[index].substring(2);
      if (!OPTIONS.contains(name)) {
        throw new IllegalArgumentException("unknown comparison option: --" + name);
      }
      if (values.put(name, args[index + 1]) != null) {
        throw new IllegalArgumentException("duplicate comparison option: --" + name);
      }
    }
    return values;
  }

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }
}
