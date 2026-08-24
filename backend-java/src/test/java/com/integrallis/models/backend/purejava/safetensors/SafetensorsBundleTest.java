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
package com.integrallis.models.backend.purejava.safetensors;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class SafetensorsBundleTest {

  private static final String BASE = "model.language_model.layers.49.mlp.down_proj";
  private static final String PACKED = BASE + ".weight_packed";
  private static final String SCALE = BASE + ".weight_scale";
  private static final String GLOBAL = BASE + ".weight_global_scale";
  private static final String INPUT = BASE + ".input_global_scale";

  @Test
  void resolvesEveryNvfp4PartThroughTheIndexAcrossShardBoundaries(@TempDir Path directory)
      throws IOException {
    String first = "model-00001-of-00002.safetensors";
    String second = "model-00002-of-00002.safetensors";
    Files.write(
        directory.resolve(first),
        new SyntheticSafetensorsBuilder()
            .add(PACKED, "U8", new long[] {2, 2}, 0x01, 0x23, 0x45, 0x67)
            .build());
    Files.write(
        directory.resolve(second),
        new SyntheticSafetensorsBuilder()
            .add(SCALE, "F8_E4M3", new long[] {2, 1}, 0x38, 0x40)
            .add(GLOBAL, "F32", new long[] {1}, 0, 0, (byte) 0x80, 0x40)
            .add(INPUT, "F32", new long[] {1}, 0, 0, (byte) 0x80, 0x3f)
            .build());
    writeIndex(
        directory,
        "{\"metadata\":{\"total_size\":14},\"weight_map\":{"
            + quote(PACKED)
            + ":\""
            + first
            + "\","
            + quote(SCALE)
            + ":\""
            + second
            + "\","
            + quote(GLOBAL)
            + ":\""
            + second
            + "\","
            + quote(INPUT)
            + ":\""
            + second
            + "\"}}");

    try (Arena arena = Arena.ofConfined()) {
      SafetensorsBundle bundle = SafetensorsBundle.open(directory, arena);

      assertThat(bundle.tensorNames()).containsExactly(PACKED, SCALE, GLOBAL, INPUT);
      assertThat(bundle.shardPath(PACKED).getFileName().toString()).isEqualTo(first);
      assertThat(bundle.shardPath(SCALE).getFileName().toString()).isEqualTo(second);
      assertThat(bundle.tensor(PACKED).data().toArray(ValueLayout.JAVA_BYTE))
          .containsExactly(0x01, 0x23, 0x45, 0x67);
      assertThat(bundle.tensor(SCALE).data().toArray(ValueLayout.JAVA_BYTE))
          .containsExactly(0x38, 0x40);
    }
  }

  @Test
  void rejectsUnsafeShardPathsAndIndexShardDisagreement(@TempDir Path directory)
      throws IOException {
    writeIndex(directory, "{\"weight_map\":{\"weight\":\"../outside.safetensors\"}}");

    try (Arena arena = Arena.ofConfined()) {
      assertThatThrownBy(() -> SafetensorsBundle.open(directory, arena))
          .isInstanceOf(MalformedSafetensorsException.class)
          .hasMessageContaining("root shard filename");
    }

    Files.write(
        directory.resolve("model-00001-of-00001.safetensors"),
        new SyntheticSafetensorsBuilder().add("different", "U8", new long[] {1}, 7).build());
    writeIndex(directory, "{\"weight_map\":{\"weight\":\"model-00001-of-00001.safetensors\"}}");

    try (Arena arena = Arena.ofConfined()) {
      assertThatThrownBy(() -> SafetensorsBundle.open(directory, arena))
          .isInstanceOf(MalformedSafetensorsException.class)
          .hasMessageContaining("weight")
          .hasMessageContaining("does not contain");
    }
  }

  @Test
  void opensAStandardSingleFileBundle(@TempDir Path directory) throws IOException {
    Path model = directory.resolve("model.safetensors");
    Files.write(
        model, new SyntheticSafetensorsBuilder().add("weight", "U8", new long[] {2}, 3, 4).build());

    try (Arena arena = Arena.ofConfined()) {
      SafetensorsBundle bundle = SafetensorsBundle.open(directory, arena);

      assertThat(bundle.tensorNames()).containsExactly("weight");
      assertThat(bundle.shardPath("weight")).isEqualTo(model.toAbsolutePath().normalize());
    }
  }

  private static void writeIndex(Path directory, String json) throws IOException {
    Files.writeString(directory.resolve("model.safetensors.index.json"), json);
  }

  private static String quote(String value) {
    return "\"" + value + "\"";
  }
}
