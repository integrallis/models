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
package com.integrallis.models.backend.tornado;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

/**
 * The arithmetic behind where the KV cache lives.
 *
 * <p>These are not measurements of a device. They are the byte counts the placement decision rests
 * on, made executable so the decision can be re-derived for any geometry rather than repeated as a
 * claim. What they cannot say is how fast a device moves those bytes; that is the GPU gate's job.
 */
class TornadoAttentionShapeTest {

  /**
   * The geometry the standing device attention experiment is built on ({@code
   * CausalAttentionExperiment}: 16 heads, 8 KV heads, head length 64), at a 4,096 context.
   */
  private static TornadoAttentionShape experimentGeometry() {
    return new TornadoAttentionShape(1, 64, 16, 8, 64, 64, 4096, 0, 0.125f);
  }

  /**
   * An illustrative larger decoder — 32 query heads over 8 KV heads at head length 128, 40 layers,
   * 8,192 context. Chosen to show the shape of the curve at a size worth accelerating; it is not a
   * transcription of any particular model's metadata.
   */
  private static TornadoAttentionShape largeGeometry() {
    return new TornadoAttentionShape(1, 64, 32, 8, 128, 128, 8192, 0, 0.088388f);
  }

  @Test
  void aMirroredPositionCostsKeyAndValueWidthInFloats() {
    TornadoAttentionShape shape = experimentGeometry();

    // 8 KV heads * 64 = 512 keys + 512 values, 4 bytes each.
    assertThat(shape.keyDim()).isEqualTo(512);
    assertThat(shape.valueDim()).isEqualTo(512);
    assertThat(shape.mirrorBytesPerPosition()).isEqualTo(4096L);
    assertThat(shape.mirrorBytesPerLayer()).isEqualTo(4096L * 4096L);
  }

  @Test
  void hostResidentUploadGrowsWithPositionAndMirroredUploadDoesNot() {
    TornadoAttentionShape shape = largeGeometry();
    int layers = 40;

    long mirrored = shape.mirroredUploadBytesPerDecodeStep(layers);
    long hostAtEight = shape.hostResidentUploadBytesPerDecodeStep(layers, 7);
    long hostAtOneK = shape.hostResidentUploadBytesPerDecodeStep(layers, 1023);
    long hostAtEightK = shape.hostResidentUploadBytesPerDecodeStep(layers, 8191);

    // 8 KV heads * 128 = 1024 keys + 1024 values * 4 bytes = 8 KiB per position per layer.
    assertThat(shape.mirrorBytesPerPosition()).isEqualTo(8192L);
    // Mirrored: one new K/V row, one query block, one output block, per layer.
    assertThat(mirrored).isEqualTo(40L * (8192L + (4096L + 4096L) * 4L));
    assertThat(hostAtEight).isEqualTo(40L * 8192L * 8L);
    assertThat(hostAtOneK).isEqualTo(40L * 8192L * 1024L);
    assertThat(hostAtEightK).isEqualTo(40L * 8192L * 8192L);

    // The point of the decision: at a full context the host-resident variant moves more than a
    // gibibyte for one token, and the mirrored variant moves the same ~1.6 MiB it moves at
    // position zero.
    assertThat(hostAtEightK).isGreaterThan(2L * 1024 * 1024 * 1024 / 2);
    assertThat(mirrored).isLessThan(2L * 1024 * 1024);
    assertThat((double) hostAtEightK / mirrored).isGreaterThan(800.0);
  }

  @Test
  void aHostResidentCacheIsTheMoreExpensiveOptionFromTheSecondTokenOnward() {
    TornadoAttentionShape shape = largeGeometry();
    int layers = 40;
    // Both placements upload the query block and download the output block, so the honest
    // comparison is over the KV bytes alone: the whole attended window every step against one new
    // row every step.
    long mirroredKvPerStep = shape.mirrorBytesPerPosition() * layers;

    int crossover = 0;
    while (shape.hostResidentUploadBytesPerDecodeStep(layers, crossover) <= mirroredKvPerStep) {
      crossover++;
    }

    assertThat(crossover).isEqualTo(1);
    assertThat(shape.hostResidentUploadBytesPerDecodeStep(layers, 0)).isEqualTo(mirroredKvPerStep);
  }

  @Test
  void aSlidingWindowBoundsWhatAHostResidentCacheWouldHaveToUpload() {
    TornadoAttentionShape windowed =
        new TornadoAttentionShape(1, 64, 32, 8, 128, 128, 8192, 512, 0.1f);

    assertThat(windowed.hostResidentUploadBytesPerDecodeStep(40, 8191))
        .isEqualTo(40L * 8192L * 512L);
  }

  @Test
  void theScoreScratchIsSizedForTheWholeContextBecauseAPlanIsCompiledOnce() {
    TornadoAttentionShape decode = largeGeometry();

    // 1 row * 32 heads * 8192 positions * 4 bytes.
    assertThat(decode.scratchBytesPerLayer()).isGreaterThan(32L * 8192L * 4L);
  }

  @Test
  void totalDeviceBytesCoverEveryLayersMirrorAndScratch() {
    TornadoAttentionShape shape = largeGeometry();

    assertThat(shape.deviceBytes(40))
        .isEqualTo(40L * (shape.mirrorBytesPerLayer() + shape.scratchBytesPerLayer()));
  }

  @Test
  void aPrefillShapeCostsTheSameMirrorAndMoreScratchThanADecodeShape() {
    TornadoAttentionShape decode = new TornadoAttentionShape(1, 64, 32, 8, 128, 128, 4096, 0, 0.1f);
    TornadoAttentionShape prefill =
        new TornadoAttentionShape(32, 64, 32, 8, 128, 128, 4096, 0, 0.1f);

    assertThat(prefill.mirrorBytesPerLayer()).isEqualTo(decode.mirrorBytesPerLayer());
    assertThat(prefill.scratchBytesPerLayer()).isGreaterThan(decode.scratchBytesPerLayer());
  }

  @Test
  void rejectsHeadGeometryThatGroupedQueryAttentionCannotExpress() {
    assertThatThrownBy(() -> new TornadoAttentionShape(1, 1, 7, 2, 64, 64, 128, 0, 0.1f))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("numKvHeads");
  }

  @Test
  void rejectsAStagingBufferSmallerThanTheExecutionBatch() {
    assertThatThrownBy(() -> new TornadoAttentionShape(32, 8, 8, 8, 64, 64, 128, 0, 0.1f))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("stagingRows");
  }

  @Test
  void rejectsAContextShorterThanOneExecution() {
    assertThatThrownBy(() -> new TornadoAttentionShape(32, 64, 8, 8, 64, 64, 16, 0, 0.1f))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("maxSequenceLength");
  }

  @Test
  void rejectsANonFiniteAttentionScale() {
    assertThatThrownBy(() -> new TornadoAttentionShape(1, 1, 8, 8, 64, 64, 128, 0, Float.NaN))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("scale");
  }

  @Test
  void rejectsNonPositiveLayerCounts() {
    assertThatThrownBy(() -> experimentGeometry().deviceBytes(0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("numLayers");
    assertThatThrownBy(() -> experimentGeometry().mirroredUploadBytesPerDecodeStep(0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("numLayers");
    assertThatThrownBy(() -> experimentGeometry().hostResidentUploadBytesPerDecodeStep(1, -1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("position");
  }
}
