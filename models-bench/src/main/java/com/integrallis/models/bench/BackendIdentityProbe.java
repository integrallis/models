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

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Validates that an HTTP backend is serving the exact GGUF bytes named by the benchmark. */
final class BackendIdentityProbe {

  private static final Pattern OLLAMA_BLOB =
      Pattern.compile("(?im)^\\s*FROM\\s+([^\\r\\n]*?sha256[:_-]([0-9a-f]{64}))\\s*$");
  private static final Pattern OLLAMA_ADAPTER = Pattern.compile("(?im)^\\s*ADAPTER\\s+");

  private BackendIdentityProbe() {}

  static BackendIdentity fromOllama(
      String model, BenchmarkArtifact expected, JsonNode showResponse, String version) {
    String modelfile = showResponse.path("modelfile").asText("");
    if (OLLAMA_ADAPTER.matcher(modelfile).find()) {
      throw new IllegalArgumentException(
          "Ollama model contains an ADAPTER and is not the same weight artifact");
    }
    Matcher from = OLLAMA_BLOB.matcher(modelfile);
    if (!from.find()) {
      throw new IllegalArgumentException(
          "Ollama model does not expose a SHA-256 GGUF blob in its FROM directive");
    }
    String runtimeSha = from.group(2).toLowerCase(Locale.ROOT);
    requireMatchingDigest("Ollama", expected.sha256(), runtimeSha);
    return new BackendIdentity(
        "ollama",
        unknownIfBlank(version),
        "unknown",
        model,
        from.group(1),
        runtimeSha,
        true,
        Map.of());
  }

  static BackendIdentity fromLlamaCpp(
      BenchmarkArtifact expected, JsonNode props, int expectedContextLength) throws IOException {
    String modelPath = props.path("model_path").asText("");
    if (modelPath.isBlank()) {
      throw new IllegalArgumentException("llama.cpp /props did not expose model_path");
    }
    int runtimeContext =
        props.path("default_generation_settings").path("n_ctx").asInt(-1);
    if (runtimeContext != expectedContextLength) {
      throw new IllegalArgumentException(
          "llama.cpp context length "
              + runtimeContext
              + " does not match benchmark context length "
              + expectedContextLength);
    }

    Path runtimePath = Path.of(modelPath);
    if (!Files.isRegularFile(runtimePath)) {
      throw new IllegalArgumentException(
          "cannot verify llama.cpp model_path from this host; start llama-server with an absolute, "
              + "locally readable path: "
              + modelPath);
    }
    BenchmarkArtifact runtime = BenchmarkArtifact.inspect(runtimePath);
    requireMatchingDigest("llama.cpp", expected.sha256(), runtime.sha256());
    String build = unknownIfBlank(props.path("build_info").asText(""));
    return new BackendIdentity(
        "llama.cpp",
        build,
        build,
        modelPath,
        runtime.path(),
        runtime.sha256(),
        true,
        Map.of("contextLength", Integer.toString(runtimeContext)));
  }

  private static void requireMatchingDigest(String backend, String expected, String actual) {
    if (!expected.equals(actual)) {
      throw new IllegalArgumentException(
          backend
              + " runtime artifact SHA-256 "
              + actual
              + " does not match compared artifact "
              + expected);
    }
  }

  private static String unknownIfBlank(String value) {
    return value == null || value.isBlank() ? "unknown" : value;
  }
}
