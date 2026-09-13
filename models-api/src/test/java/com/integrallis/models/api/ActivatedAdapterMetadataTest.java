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
package com.integrallis.models.api;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class ActivatedAdapterMetadataTest {

  @Test
  void normalizesHashesAndDefensivelyCopiesTheActivationSequence() {
    List<Integer> invocation = new ArrayList<>(List.of(151644, 77091, 198));

    ActivatedAdapterMetadata metadata =
        new ActivatedAdapterMetadata(
            "Qwen/Qwen3-0.6B",
            "A".repeat(40),
            "B".repeat(64),
            Map.of("tokenizer.json", "D".repeat(64)),
            "C".repeat(64),
            32,
            64,
            invocation,
            provenance());
    invocation.clear();

    assertThat(metadata.baseRevision()).isEqualTo("a".repeat(40));
    assertThat(metadata.baseArtifactSha256()).isEqualTo("b".repeat(64));
    assertThat(metadata.adapterSha256()).isEqualTo("c".repeat(64));
    assertThat(metadata.tokenizerFileSha256())
        .containsExactly(Map.entry("tokenizer.json", "d".repeat(64)));
    assertThat(metadata.trainingProvenance().datasetRevision()).isEqualTo("e".repeat(40));
    assertThat(metadata.invocationTokens()).containsExactly(151644, 77091, 198);
    assertThatThrownBy(() -> metadata.invocationTokens().add(1))
        .isInstanceOf(UnsupportedOperationException.class);
  }

  @Test
  void rejectsUnpinnedOrMalformedProvenance() {
    assertThatThrownBy(
            () ->
                new ActivatedAdapterMetadata(
                    "base",
                    "main",
                    "b".repeat(64),
                    tokenizers(),
                    "c".repeat(64),
                    1,
                    1,
                    List.of(1),
                    provenance()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("baseRevision");
    assertThatThrownBy(
            () ->
                new ActivatedAdapterMetadata(
                    "base",
                    "a".repeat(40),
                    "sha",
                    tokenizers(),
                    "c".repeat(64),
                    1,
                    1,
                    List.of(1),
                    provenance()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("baseArtifactSha256");
    assertThatThrownBy(
            () ->
                new ActivatedAdapterMetadata(
                    "base",
                    "a".repeat(40),
                    "b".repeat(64),
                    tokenizers(),
                    "c".repeat(64),
                    0,
                    1,
                    List.of(1),
                    provenance()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("rank");
    assertThatThrownBy(
            () ->
                new ActivatedAdapterMetadata(
                    "base",
                    "a".repeat(40),
                    "b".repeat(64),
                    tokenizers(),
                    "c".repeat(64),
                    1,
                    0,
                    List.of(1),
                    provenance()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("alpha");
    assertThatThrownBy(
            () ->
                new ActivatedAdapterMetadata(
                    "base",
                    "a".repeat(40),
                    "b".repeat(64),
                    tokenizers(),
                    "c".repeat(64),
                    1,
                    1,
                    List.of(),
                    provenance()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("invocationTokens");
    assertThatThrownBy(
            () ->
                new ActivatedAdapterMetadata(
                    "base",
                    "a".repeat(40),
                    "b".repeat(64),
                    Map.of(),
                    "c".repeat(64),
                    1,
                    1,
                    List.of(1),
                    provenance()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("tokenizerFileSha256");
    assertThatThrownBy(
            () ->
                new ActivatedAdapterMetadata(
                    "base",
                    "a".repeat(40),
                    "b".repeat(64),
                    tokenizers(),
                    "c".repeat(64),
                    1,
                    1,
                    java.util.Arrays.asList(1, null),
                    provenance()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("invocationTokens");
  }

  @Test
  void preservesEveryPinnedTrainingSource() {
    var sources =
        new ArrayList<ActivatedAdapterMetadata.TrainingSource>(
            List.of(
                new ActivatedAdapterMetadata.TrainingSource(
                    "tool-calls",
                    "edbuildingstuff/bfcl-ft-data",
                    "e".repeat(40),
                    "train.javajs.messages.jsonl",
                    "f".repeat(64)),
                new ActivatedAdapterMetadata.TrainingSource(
                    "no-call",
                    "MadeAgents/xlam-irrelevance-7.5k",
                    "5".repeat(40),
                    "xlam-7.5k-irrelevancek.json",
                    "6".repeat(64))));
    var provenance =
        new ActivatedAdapterMetadata.TrainingProvenance(
            sources,
            2,
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "train_alora.py",
            "4".repeat(64));
    sources.clear();

    assertThat(provenance.sources()).hasSize(2);
    assertThat(provenance.dataset()).isEqualTo("edbuildingstuff/bfcl-ft-data");
    assertThat(provenance.sources().get(1).role()).isEqualTo("no-call");
    assertThat(provenance.preparedSchemaVersion()).isEqualTo(2);
    assertThat(provenance.trainSelection()).isEmpty();
    assertThatThrownBy(() -> provenance.sources().add(provenance.sources().getFirst()))
        .isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(
            () ->
                new ActivatedAdapterMetadata.TrainingProvenance(
                    List.of(sourcesForDuplicateCheck(), sourcesForDuplicateCheck()),
                    2,
                    "1".repeat(64),
                    "2".repeat(64),
                    "3".repeat(64),
                    "train_alora.py",
                    "4".repeat(64)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("duplicate");
  }

  @Test
  void preservesTheExactRowsSelectedFromPreparedSplits() {
    var train = new ActivatedAdapterMetadata.TrainingSelection(4000, 64, 760, 1709, "7".repeat(64));
    var validation =
        new ActivatedAdapterMetadata.TrainingSelection(979, 21, 190, 420, "8".repeat(64));
    var provenance =
        new ActivatedAdapterMetadata.TrainingProvenance(
            List.of(sourcesForDuplicateCheck()),
            2,
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            Optional.of(train),
            Optional.of(validation),
            "train_alora.py",
            "4".repeat(64));

    assertThat(provenance.trainSelection()).contains(train);
    assertThat(provenance.validationSelection()).contains(validation);
    assertThatThrownBy(
            () -> new ActivatedAdapterMetadata.TrainingSelection(10, 0, 11, 0, "7".repeat(64)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("noCall");
  }

  private static ActivatedAdapterMetadata.TrainingSource sourcesForDuplicateCheck() {
    return new ActivatedAdapterMetadata.TrainingSource(
        "tool-calls",
        "edbuildingstuff/bfcl-ft-data",
        "e".repeat(40),
        "train.javajs.messages.jsonl",
        "f".repeat(64));
  }

  private static Map<String, String> tokenizers() {
    return Map.of("tokenizer.json", "d".repeat(64));
  }

  private static ActivatedAdapterMetadata.TrainingProvenance provenance() {
    return new ActivatedAdapterMetadata.TrainingProvenance(
        "edbuildingstuff/bfcl-ft-data",
        "e".repeat(40),
        "train.javajs.messages.jsonl",
        "f".repeat(64),
        "1".repeat(64),
        "2".repeat(64),
        "3".repeat(64),
        "train_alora.py",
        "4".repeat(64));
  }
}
