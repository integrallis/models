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
package com.integrallis.models.backend.purejava.gguf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GgufEmbeddedFilesTest {

  @Test
  void readsBinarySidecarsFromAudioCppGgufMetadata() {
    byte[] config = "{\"model_type\":\"qwen3\"}".getBytes(StandardCharsets.UTF_8);
    byte[] tokenizer = "{\"version\":\"1.0\"}".getBytes(StandardCharsets.UTF_8);
    byte[] payload = new byte[config.length + tokenizer.length];
    System.arraycopy(config, 0, payload, 0, config.length);
    System.arraycopy(tokenizer, 0, payload, config.length, tokenizer.length);
    byte[] gguf =
        new SyntheticGgufBuilder()
            .addStringArray(
                "audiocpp.embedded_files.names", List.of("config.json", "tokenizer.json"))
            .addUint64Array(
                "audiocpp.embedded_files.offsets", new long[] {0, config.length, payload.length})
            .addUint8Array("audiocpp.embedded_files.data", payload)
            .build();

    GgufFile file = GgufParser.parseSegment(MemorySegment.ofArray(gguf));
    GgufEmbeddedFiles embedded = GgufEmbeddedFiles.from(file.metadata());

    assertThat(embedded.names()).containsExactly("config.json", "tokenizer.json");
    assertThat(embedded.readUtf8("config.json")).isEqualTo("{\"model_type\":\"qwen3\"}");
    byte[] returned = embedded.read("tokenizer.json");
    returned[0] = 0;
    assertThat(embedded.readUtf8("tokenizer.json")).isEqualTo("{\"version\":\"1.0\"}");
  }

  @Test
  void rejectsUnsafePathsAndInvalidRanges() {
    assertThatThrownBy(
            () ->
                GgufEmbeddedFiles.from(
                    GgufParser.parseSegment(
                            MemorySegment.ofArray(
                                new SyntheticGgufBuilder()
                                    .addStringArray(
                                        "audiocpp.embedded_files.names", List.of("../secret"))
                                    .addUint64Array(
                                        "audiocpp.embedded_files.offsets", new long[] {0, 1})
                                    .addUint8Array("audiocpp.embedded_files.data", new byte[] {1})
                                    .build()))
                        .metadata()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("unsafe");

    assertThatThrownBy(
            () ->
                GgufEmbeddedFiles.from(
                    GgufParser.parseSegment(
                            MemorySegment.ofArray(
                                new SyntheticGgufBuilder()
                                    .addStringArray(
                                        "audiocpp.embedded_files.names", List.of("config.json"))
                                    .addUint64Array(
                                        "audiocpp.embedded_files.offsets", new long[] {0, 2})
                                    .addUint8Array("audiocpp.embedded_files.data", new byte[] {1})
                                    .build()))
                        .metadata()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("range");
  }
}
