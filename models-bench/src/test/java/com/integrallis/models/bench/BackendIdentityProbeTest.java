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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BackendIdentityProbeTest {

  private final ObjectMapper mapper = new ObjectMapper();

  @TempDir Path directory;

  @Test
  void verifiesAnOllamaModelBackedByTheExactGgufBlob() throws Exception {
    Path artifact = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});
    BenchmarkArtifact expected = BenchmarkArtifact.inspect(artifact);
    String showResponse =
        mapper
            .createObjectNode()
            .put(
                "modelfile",
                "FROM /var/lib/ollama/models/blobs/sha256:" + expected.sha256() + "\n")
            .toString();

    BackendIdentity identity =
        BackendIdentityProbe.fromOllama(
            "fixture:latest", expected, mapper.readTree(showResponse), "0.12.1");

    assertThat(identity.artifactVerified()).isTrue();
    assertThat(identity.runtimeArtifactSha256()).isEqualTo(expected.sha256());
    assertThat(identity.version()).isEqualTo("0.12.1");
  }

  @Test
  void rejectsAnOllamaModelWhoseBlobDoesNotMatchTheComparedArtifact() throws Exception {
    Path artifact = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});
    BenchmarkArtifact expected = BenchmarkArtifact.inspect(artifact);
    String otherSha = "0".repeat(64);

    assertThatThrownBy(
            () ->
                BackendIdentityProbe.fromOllama(
                    "fixture:latest",
                    expected,
                    mapper.readTree("{\"modelfile\":\"FROM sha256:" + otherSha + "\\n\"}"),
                    "0.12.1"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("does not match");
  }

  @Test
  void rejectsOllamaAdaptersBecauseTheyChangeTheComparedWeights() throws Exception {
    Path artifact = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});
    BenchmarkArtifact expected = BenchmarkArtifact.inspect(artifact);
    String modelfile =
        "FROM /var/lib/ollama/models/blobs/sha256:"
            + expected.sha256()
            + "\nADAPTER /tmp/adapter.gguf\n";

    assertThatThrownBy(
            () ->
                BackendIdentityProbe.fromOllama(
                    "fixture:latest",
                    expected,
                    mapper.createObjectNode().put("modelfile", modelfile),
                    "0.12.1"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("ADAPTER");
  }

  @Test
  void verifiesLlamaCppModelPathAndContextAgainstLocalBytes() throws Exception {
    Path artifact = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});
    BenchmarkArtifact expected = BenchmarkArtifact.inspect(artifact);
    String props =
        mapper
            .createObjectNode()
            .put("model_path", artifact.toString())
            .put("build_info", "b8123-a1b2c3d")
            .set(
                "default_generation_settings",
                mapper.createObjectNode().put("n_ctx", 2_048))
            .toString();

    BackendIdentity identity =
        BackendIdentityProbe.fromLlamaCpp(expected, mapper.readTree(props), 2_048);

    assertThat(identity.artifactVerified()).isTrue();
    assertThat(identity.runtimeArtifactSha256()).isEqualTo(expected.sha256());
    assertThat(identity.build()).isEqualTo("b8123-a1b2c3d");
  }

  @Test
  void rejectsLlamaCppRunningWithADifferentContext() throws Exception {
    Path artifact = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});
    BenchmarkArtifact expected = BenchmarkArtifact.inspect(artifact);
    String props =
        mapper
            .createObjectNode()
            .put("model_path", artifact.toString())
            .set(
                "default_generation_settings",
                mapper.createObjectNode().put("n_ctx", 4_096))
            .toString();

    assertThatThrownBy(
            () -> BackendIdentityProbe.fromLlamaCpp(expected, mapper.readTree(props), 2_048))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("context length");
  }
}
