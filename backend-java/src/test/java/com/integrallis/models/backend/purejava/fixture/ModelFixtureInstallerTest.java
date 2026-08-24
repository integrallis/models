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
package com.integrallis.models.backend.purejava.fixture;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelFixtureInstallerTest {
  @TempDir Path temporaryDirectory;

  @Test
  void installsVerifiedArtifactFromPinnedUri() throws Exception {
    byte[] content = "verified fixture".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    Path source = temporaryDirectory.resolve("source.gguf");
    Path target = temporaryDirectory.resolve("installed/model.gguf");
    Files.write(source, content);

    Path installed = ModelFixtureInstaller.install(descriptor(source, target, content));

    assertThat(installed).isEqualTo(target);
    assertThat(Files.readAllBytes(target)).isEqualTo(content);
  }

  @Test
  void rejectsInvalidDownloadWithoutReplacingExistingArtifact() throws Exception {
    byte[] existing = "existing".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    byte[] downloaded = "downloaded".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    Path source = temporaryDirectory.resolve("source.gguf");
    Path target = temporaryDirectory.resolve("installed/model.gguf");
    Files.createDirectories(target.getParent());
    Files.write(source, downloaded);
    Files.write(target, existing);
    ModelFixtureDescriptor descriptor =
        descriptor(source, target, "different".getBytes(java.nio.charset.StandardCharsets.UTF_8));

    assertThatThrownBy(() -> ModelFixtureInstaller.install(descriptor))
        .isInstanceOf(IOException.class)
        .hasMessageContaining("mismatch");
    assertThat(Files.readAllBytes(target)).isEqualTo(existing);
  }

  private static ModelFixtureDescriptor descriptor(Path source, Path target, byte[] expected)
      throws Exception {
    return new ModelFixtureDescriptor(
        "fixture",
        "Fixture",
        "1.0.0",
        "file://fixture",
        "f32",
        "pure-java",
        source.toUri(),
        target,
        sha256(expected),
        expected.length,
        "test",
        "F32",
        Set.of("text-generation"),
        Set.of(),
        false,
        "gguf");
  }

  private static String sha256(byte[] content) throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(content));
  }
}
