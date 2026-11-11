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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/**
 * Native attention retained-state serialization round-trip.
 *
 * <p>Exported state must restore exact layer identity and F16 bit patterns so a receiver
 * continues from the same positions without recompute.
 */
class NativeCausalAttentionStateTest {

  private static final int HEADS = 2;
  private static final int WIDTH = 64;
  private static final int CAPACITY = 128;
  private static final int DIMENSION = HEADS * WIDTH;

  /** Pre-condition: the native kernel library must be present. */
  private static Path requireLibrary() {
    String configured = System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY);
    assumeTrue(configured != null, RustFfmBackend.LIBRARY_PATH_PROPERTY + " is not set");
    Path library = Path.of(configured);
    assumeTrue(Files.isRegularFile(library), library + " is not a file");
    return library;
  }

  /** Populate keys/values with deterministic floats so rounding is reproducible. */
  private static void populate(float[] k, float[] v, float kBase, float vBase, float scale) {
    for (int i = 0; i < k.length; i++) {
      k[i] = kBase + i * scale;
      v[i] = vBase - i * scale;
    }
  }

  /** Write the current retained state to a byte array and return it. */
  private static byte[] export(NativeCausalAttention attention) throws Exception {
    var output = new ByteArrayOutputStream();
    attention.writeRetainedState(new DataOutputStream(output));
    byte[] bytes = output.toByteArray();
    assertThat(bytes.length).isEqualTo(attention.retainedStateBytes());
    return bytes;
  }

  /** Perform an attend on the given layer starting at position 0 with {@code batch} tokens. */
  private static void attendInitial(
      NativeCausalAttention attention, int layer, int batch) {
    float[] q = new float[batch * DIMENSION];
    float[] k = new float[batch * DIMENSION];
    float[] v = new float[batch * DIMENSION];
    float[] out = new float[batch * DIMENSION];
    populate(k, v, 1.0f + layer, -1.0f - layer, 0.01f);
    attention.attend(out, q, k, v, layer, 0, batch, HEADS, HEADS, WIDTH, WIDTH, CAPACITY, 0);
  }

  @Test
  void emptyDonorExportsZeroLengthAndReceiverImportCreatesNoLayers() throws Exception {
    Path library = requireLibrary();
    try (var donor = new NativeCausalAttention(NativeDebertaKernels.load(library, 3));
        var receiver = new NativeCausalAttention(NativeDebertaKernels.load(library, 3))) {
      assertThat(donor.supportsPortableState()).isTrue();
      assertThat(donor.retainedStateBytes()).isEqualTo(Integer.BYTES);

      receiver.readRetainedState(new DataInputStream(new ByteArrayInputStream(export(donor))));

      assertThat(receiver.allocatedStateBytes()).isEqualTo(donor.allocatedStateBytes()).isZero();
    }
  }

  @Test
  void importedStatePreservesLayerCountAndAllocatedBytes() throws Exception {
    Path library = requireLibrary();
    try (var dense = NativeDebertaKernels.load(library, 3);
        var donor = new NativeCausalAttention(dense);
        var receiver = new NativeCausalAttention(NativeDebertaKernels.load(library, 3))) {
      // Pre-condition: donor has two layers with different batch sizes.
      attendInitial(donor, 0, 4); // layer 0, 4 positions
      attendInitial(donor, 5, 3); // layer 5, 3 positions
      long donorBytes = donor.allocatedStateBytes();
      assertThat(donorBytes).isPositive();

      receiver.readRetainedState(new DataInputStream(new ByteArrayInputStream(export(donor))));

      assertThat(receiver.allocatedStateBytes()).isEqualTo(donorBytes);
    }
  }

  @Test
  void importedStateRestoresPositionsForExactContinuation() throws Exception {
    Path library = requireLibrary();
    try (var dense = NativeDebertaKernels.load(library, 3);
        var donor = new NativeCausalAttention(dense);
        var receiver = new NativeCausalAttention(NativeDebertaKernels.load(library, 3))) {
      // Pre-condition: donor has attended 4 tokens on layer 0.
      attendInitial(donor, 0, 4);

      receiver.readRetainedState(new DataInputStream(new ByteArrayInputStream(export(donor))));

      // Post-condition: receiver must accept continuation at position 4 (where donor stopped).
      assertThat(receiver.isEligible(0, 4, 2, HEADS, HEADS, WIDTH, WIDTH, CAPACITY, 0))
          .isTrue();
      // Sanity: donor must still accept the same continuation.
      assertThat(donor.isEligible(0, 4, 2, HEADS, HEADS, WIDTH, WIDTH, CAPACITY, 0))
          .isTrue();
    }
  }

  @Test
  void importedStateProducesIdenticalAttendOutputWhenContinuing() throws Exception {
    Path library = requireLibrary();
    try (var denseDonor = NativeDebertaKernels.load(library, 3);
        var denseContinue = NativeDebertaKernels.load(library, 3);
        var denseRecv = NativeDebertaKernels.load(library, 3);
        var donorFirst = new NativeCausalAttention(denseDonor);
        var donorContinue = new NativeCausalAttention(denseContinue);
        var receiver = new NativeCausalAttention(denseRecv)) {

      // Phase 1: both donorFirst and donorContinue establish identical prefix state.
      float[] q1 = new float[4 * DIMENSION];
      float[] k1 = new float[4 * DIMENSION];
      float[] v1 = new float[4 * DIMENSION];
      float[] out1 = new float[4 * DIMENSION];
      populate(k1, v1, 1.0f, -1.0f, 0.01f);
      donorFirst.attend(out1, q1, k1, v1, 0, 0, 4, HEADS, HEADS, WIDTH, WIDTH, CAPACITY, 0);
      donorContinue.attend(out1.clone(), q1.clone(), k1.clone(), v1.clone(), 0, 0, 4, HEADS, HEADS, WIDTH, WIDTH, CAPACITY, 0);

      // Phase 2: export donorFirst's state after the prefix.
      byte[] serialized = export(donorFirst);

      // Phase 3a: donorContinue continues with 2 more tokens from position 4.
      float[] q2 = new float[2 * DIMENSION];
      float[] k2 = new float[2 * DIMENSION];
      float[] v2 = new float[2 * DIMENSION];
      float[] donorOut = new float[2 * DIMENSION];
      populate(k2, v2, 0.5f, -0.5f, 0.005f);
      donorContinue.attend(donorOut, q2, k2, v2, 0, 4, 2, HEADS, HEADS, WIDTH, WIDTH, CAPACITY, 0);

      // Phase 3b: receiver imports prefix state, then continues with the same 2 tokens.
      receiver.readRetainedState(new DataInputStream(new ByteArrayInputStream(serialized)));
      float[] recvOut = new float[2 * DIMENSION];
      receiver.attend(recvOut, q2.clone(), k2.clone(), v2.clone(), 0, 4, 2, HEADS, HEADS, WIDTH, WIDTH, CAPACITY, 0);

      // Phase 4: exact-token continuation must produce identical F16-rounded outputs.
      assertThat(recvOut).containsExactly(donorOut);
    }
  }

  @Test
  void importClobbersExistingReceiverLayers() throws Exception {
    Path library = requireLibrary();
    try (var dense = NativeDebertaKernels.load(library, 3);
        var donor = new NativeCausalAttention(dense);
        var receiver = new NativeCausalAttention(NativeDebertaKernels.load(library, 3))) {
      // Pre-condition: receiver already holds unrelated layer 1 data.
      attendInitial(receiver, 1, 5);
      long recvBefore = receiver.allocatedStateBytes();
      assertThat(recvBefore).isPositive();

      // donor exports its own state.
      attendInitial(donor, 0, 4);
      long donorBytes = donor.allocatedStateBytes();

      // Post-condition: import replaces, not accumulates.
      receiver.readRetainedState(new DataInputStream(new ByteArrayInputStream(export(donor))));
      assertThat(receiver.allocatedStateBytes()).isEqualTo(donorBytes);
    }
  }
}
