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
package com.integrallis.models.backend.purejava.lora;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.lora.ActivatedLoraAdapter.Architecture;
import com.integrallis.models.backend.purejava.lora.ActivatedLoraAdapter.Projection;
import com.integrallis.models.backend.purejava.safetensors.SyntheticSafetensorsBuilder;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class ActivatedLoraAdapterTest {

  private static final String BASE_SHA = "a".repeat(64);
  private static final String REVISION = "b".repeat(40);
  private static final Architecture ARCHITECTURE = new Architecture(1, 2, 2, 1, 1, 2, 3);

  @TempDir Path temporaryDirectory;

  @Test
  void loadsAnExactPinnedAdapterAndAppliesEveryDeclaredProjection() throws Exception {
    writeAdapter(false);

    try (Arena arena = Arena.ofConfined()) {
      ActivatedLoraAdapter adapter =
          ActivatedLoraAdapter.open(temporaryDirectory, arena, BASE_SHA, ARCHITECTURE);

      assertThat(adapter.baseModel()).isEqualTo("Qwen/Qwen3-0.6B");
      assertThat(adapter.baseRevision()).isEqualTo(REVISION);
      assertThat(adapter.tokenizerFileSha256())
          .containsExactly(java.util.Map.entry("tokenizer.json", "d".repeat(64)));
      assertThat(adapter.trainingProvenance().datasetRevision()).isEqualTo("e".repeat(40));
      assertThat(adapter.trainingProvenance().sources())
          .extracting(com.integrallis.models.api.ActivatedAdapterMetadata.TrainingSource::role)
          .containsExactly("tool-calls", "no-call");
      assertThat(adapter.trainingProvenance().preparedSchemaVersion()).isEqualTo(2);
      assertThat(adapter.trainingProvenance().trainSelection()).isPresent();
      assertThat(adapter.trainingProvenance().trainSelection().orElseThrow().usable())
          .isEqualTo(4000);
      assertThat(adapter.trainingProvenance().trainSelection().orElseThrow().noCall())
          .isEqualTo(760);
      assertThat(adapter.trainingProvenance().validationSelection()).isPresent();
      assertThat(adapter.invocationTokens()).containsExactly(151644, 77091, 198);
      for (Projection projection : Projection.values()) {
        float[] output = new float[adapter.outputDimension(projection)];
        float[] input = new float[adapter.inputDimension(projection)];
        input[0] = 1.0f;
        adapter.addTo(0, projection, output, 0, input, 0);
        for (int index = 0; index < output.length; index++) {
          // A[0,0] = 0.1, B[index,0] = 0.3 + index, and alpha/rank = 2.
          assertThat(output[index]).isCloseTo(0.2f * (0.3f + index), within(1.0e-6f));
        }
      }
    }
  }

  private static org.assertj.core.data.Offset<Float> within(float tolerance) {
    return org.assertj.core.data.Offset.offset(tolerance);
  }

  @Test
  void rejectsARequestedBaseArtifactThatDiffersFromPinnedMetadata() throws Exception {
    writeAdapter(false);

    try (Arena arena = Arena.ofConfined()) {
      assertThatThrownBy(
              () ->
                  ActivatedLoraAdapter.open(
                      temporaryDirectory, arena, "c".repeat(64), ARCHITECTURE))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("base artifact SHA-256");
    }
  }

  @Test
  void rejectsAdapterContentWhoseHashOrTensorShapeDoesNotMatchTheManifest() throws Exception {
    writeAdapter(false);
    Files.writeString(temporaryDirectory.resolve("adapter_model.safetensors"), "tampered");
    try (Arena arena = Arena.ofConfined()) {
      assertThatThrownBy(
              () -> ActivatedLoraAdapter.open(temporaryDirectory, arena, BASE_SHA, ARCHITECTURE))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("SHA-256");
    }

    writeAdapter(true);
    try (Arena arena = Arena.ofConfined()) {
      assertThatThrownBy(
              () -> ActivatedLoraAdapter.open(temporaryDirectory, arena, BASE_SHA, ARCHITECTURE))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("q_proj.lora_A.weight")
          .hasMessageContaining("shape");
    }
  }

  @Test
  void rejectsLegacyOrMalformedProvenanceMetadata() throws Exception {
    writeAdapter(false);
    Path metadata = temporaryDirectory.resolve("models-activated-lora.json");
    String valid = Files.readString(metadata);

    Files.writeString(metadata, valid.replace("\"schemaVersion\": 4", "\"schemaVersion\": 1"));
    try (Arena arena = Arena.ofConfined()) {
      assertThatThrownBy(
              () -> ActivatedLoraAdapter.open(temporaryDirectory, arena, BASE_SHA, ARCHITECTURE))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("schemaVersion");
    }

    String malformedSourceHash =
        valid.replace(
            "\"sourceSha256\": \"" + "f".repeat(64) + "\"", "\"sourceSha256\": \"malformed\"");
    Files.writeString(metadata, malformedSourceHash);
    try (Arena arena = Arena.ofConfined()) {
      assertThatThrownBy(
              () -> ActivatedLoraAdapter.open(temporaryDirectory, arena, BASE_SHA, ARCHITECTURE))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("sourceSha256");
    }

    String missingSelection =
        valid.replaceFirst("(?s)\\s*\"trainSelection\"\\s*:\\s*\\{.*?\\},", "");
    Files.writeString(metadata, missingSelection);
    try (Arena arena = Arena.ofConfined()) {
      assertThatThrownBy(
              () -> ActivatedLoraAdapter.open(temporaryDirectory, arena, BASE_SHA, ARCHITECTURE))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("schema 4 requires");
    }
  }

  @Test
  void readsSchemaThreeAdaptersWithoutInventingSubsetProvenance() throws Exception {
    writeAdapter(false);
    Path metadata = temporaryDirectory.resolve("models-activated-lora.json");
    String schemaThree =
        Files.readString(metadata)
            .replace("\"schemaVersion\": 4", "\"schemaVersion\": 3")
            .replaceFirst("(?s)\\s*\"trainSelection\"\\s*:\\s*\\{.*?\\},", "")
            .replaceFirst("(?s)\\s*\"validationSelection\"\\s*:\\s*\\{.*?\\},", "");
    Files.writeString(metadata, schemaThree);

    try (Arena arena = Arena.ofConfined()) {
      ActivatedLoraAdapter adapter =
          ActivatedLoraAdapter.open(temporaryDirectory, arena, BASE_SHA, ARCHITECTURE);

      assertThat(adapter.trainingProvenance().trainSelection()).isEmpty();
      assertThat(adapter.trainingProvenance().validationSelection()).isEmpty();
    }
  }

  private void writeAdapter(boolean wrongQShape) throws Exception {
    SyntheticSafetensorsBuilder tensors = new SyntheticSafetensorsBuilder();
    addProjection(tensors, "self_attn.q_proj", wrongQShape ? 3 : 2, 2, wrongQShape);
    addProjection(tensors, "self_attn.k_proj", 2, 1, false);
    addProjection(tensors, "self_attn.v_proj", 2, 1, false);
    addProjection(tensors, "self_attn.o_proj", 2, 2, false);
    addProjection(tensors, "mlp.gate_proj", 2, 3, false);
    addProjection(tensors, "mlp.up_proj", 2, 3, false);
    addProjection(tensors, "mlp.down_proj", 3, 2, false);
    Path weights = temporaryDirectory.resolve("adapter_model.safetensors");
    Files.write(weights, tensors.build());
    String hash = sha256(weights);
    Files.writeString(
        temporaryDirectory.resolve("models-activated-lora.json"),
        """
        {
          "schemaVersion": 4,
          "kind": "activated-lora-tool-specialist",
          "base": {
            "model": "Qwen/Qwen3-0.6B",
            "revision": "%s",
            "artifactSha256": "%s"
          },
          "tokenizer": {
            "files": [
              {"name": "tokenizer.json", "sha256": "%s"}
            ]
          },
          "adapter": {
            "file": "adapter_model.safetensors",
            "sha256": "%s",
            "rank": 1,
            "alpha": 2,
            "targetModules": [
              "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ]
          },
          "invocation": {"tokens": [151644, 77091, 198]},
          "training": {
            "sources": [
              {
                "role": "tool-calls",
                "dataset": "edbuildingstuff/bfcl-ft-data",
                "datasetRevision": "%s",
                "sourceFile": "train.javajs.messages.jsonl",
                "sourceSha256": "%s"
              },
              {
                "role": "no-call",
                "dataset": "MadeAgents/xlam-irrelevance-7.5k",
                "datasetRevision": "%s",
                "sourceFile": "xlam-7.5k-irrelevancek.json",
                "sourceSha256": "%s"
              }
            ],
            "preparedSchemaVersion": 2,
            "preparedManifestSha256": "%s",
            "trainSha256": "%s",
            "validationSha256": "%s",
            "trainSelection": {
              "usable": 4000,
              "skippedOverMaxLength": 64,
              "noCall": 760,
              "multipleCall": 1709,
              "sourceLinesSha256": "%s"
            },
            "validationSelection": {
              "usable": 979,
              "skippedOverMaxLength": 21,
              "noCall": 190,
              "multipleCall": 420,
              "sourceLinesSha256": "%s"
            },
            "formatter": "train_alora.py",
            "formatterSha256": "%s"
          }
        }
        """
            .formatted(
                REVISION,
                BASE_SHA,
                "d".repeat(64),
                hash,
                "e".repeat(40),
                "f".repeat(64),
                "5".repeat(40),
                "6".repeat(64),
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "7".repeat(64),
                "8".repeat(64),
                "4".repeat(64)));
  }

  private static void addProjection(
      SyntheticSafetensorsBuilder tensors,
      String module,
      int inputDimension,
      int outputDimension,
      boolean wrongShape) {
    String prefix = "base_model.model.model.layers.0." + module;
    int aElements = inputDimension;
    int bElements = outputDimension;
    tensors.addF32(
        prefix + ".lora_A.weight", new long[] {1, inputDimension}, sequence(aElements, 0.1f));
    tensors.addF32(
        prefix + ".lora_B.weight",
        new long[] {outputDimension, 1},
        sequence(bElements, wrongShape ? 0.2f : 0.3f));
  }

  private static float[] sequence(int count, float first) {
    float[] values = new float[count];
    for (int index = 0; index < count; index++) values[index] = first + index;
    return values;
  }

  private static String sha256(Path path) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (var input = Files.newInputStream(path)) {
      byte[] buffer = new byte[8192];
      for (int read; (read = input.read(buffer)) >= 0; ) digest.update(buffer, 0, read);
    }
    return HexFormat.of().formatHex(digest.digest());
  }
}
