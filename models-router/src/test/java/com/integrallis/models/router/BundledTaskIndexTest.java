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
package com.integrallis.models.router;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Checks the index actually packaged in this jar.
 *
 * <p>A resource is easy to ship broken: the build copies whatever bytes are there, and nothing
 * downstream reads them until a user does. These open the real archive rather than a fixture, so a
 * truncated, stale or wrongly-pathed artifact fails here instead of at someone else's startup.
 *
 * <p>What they cannot check is accuracy, which needs the embedding model and therefore lives in
 * models-bench. This is about the artifact being present, intact, and describing itself honestly.
 */
@Tag("unit")
class BundledTaskIndexTest {

  /** The corpus the shipped index was built from; a rebuild against a different one must show. */
  private static final String EXPECTED_CORPUS_SHA256 =
      "639d305a2c4fee52779cb16832f365d91bea621d0908f97d501df74872121299";

  private static final int EXPECTED_PROMPTS = 1929;
  private static final int EXPECTED_DIMENSION = 768;

  @Test
  void isPackagedInTheJar() throws IOException {
    try (InputStream in = TaskIndexResource.class.getResourceAsStream(TaskIndexResource.RESOURCE)) {
      assertThat(in).describedAs("index at %s", TaskIndexResource.RESOURCE).isNotNull();
      assertThat(in.readAllBytes().length).isPositive();
    }
  }

  @Test
  void expandsAndDescribesWhatItHolds(@TempDir Path cacheRoot) throws IOException {
    Path expanded = TaskIndexResource.extractTo(cacheRoot);

    Properties manifest = new Properties();
    try (var reader = Files.newBufferedReader(expanded.resolve(TaskIndexBuilder.MANIFEST))) {
      manifest.load(reader);
    }

    assertThat(manifest.getProperty("prompts")).isEqualTo(String.valueOf(EXPECTED_PROMPTS));
    assertThat(manifest.getProperty("dimension")).isEqualTo(String.valueOf(EXPECTED_DIMENSION));
    assertThat(manifest.getProperty("corpusSha256")).isEqualTo(EXPECTED_CORPUS_SHA256);
    assertThat(manifest.getProperty("tasks").split(",")).hasSize(10);
    // Shipped as 4-bit codes with no full-precision copy. If this ever reads NONE the artifact
    // grew eightfold without anyone deciding to.
    assertThat(manifest.getProperty("quantizer")).isEqualTo("SQ4");
  }

  @Test
  void carriesCodesInsteadOfVectors(@TempDir Path cacheRoot) throws IOException {
    Path expanded = TaskIndexResource.extractTo(cacheRoot);

    try (var walk = Files.walk(expanded)) {
      long vectorBytes = 0;
      long quantizedBytes = 0;
      for (Path file : walk.filter(Files::isRegularFile).toList()) {
        if (file.getFileName().toString().equals("vectors.bin")) {
          vectorBytes += Files.size(file);
        } else if (file.getFileName().toString().equals("quantized.bin")) {
          quantizedBytes += Files.size(file);
        }
      }
      assertThat(vectorBytes).describedAs("full-precision vectors").isZero();
      assertThat(quantizedBytes).describedAs("quantized codes").isPositive();
    }
  }

  @Test
  void opensAsASearchableIndex(@TempDir Path cacheRoot) {
    try (TaskIndex index = TaskIndex.open(TaskIndexResource.extractTo(cacheRoot))) {
      assertThat(index.dimension()).isEqualTo(EXPECTED_DIMENSION);
      assertThat(index.taskNames())
          .contains("code", "sql", "math", "translation", "extraction", "chat");

      // A vector of the right width is enough to prove the index searches; what it means needs
      // the embedding model, which this module deliberately does not depend on.
      float[] probe = new float[EXPECTED_DIMENSION];
      probe[0] = 1.0f;
      assertThat(index.nearest(probe)).isPresent();
    }
  }

  @Test
  void reusesAnAlreadyExpandedCache(@TempDir Path cacheRoot) throws IOException {
    Path first = TaskIndexResource.extractTo(cacheRoot);
    // Expansion is keyed by archive digest, so a second call must land on the same directory
    // rather than unpacking a megabyte again on every startup.
    Path second = TaskIndexResource.extractTo(cacheRoot);

    assertThat(second).isEqualTo(first);
    try (var entries = Files.list(cacheRoot)) {
      assertThat(entries.filter(Files::isDirectory).toList()).hasSize(1);
    }
  }
}
