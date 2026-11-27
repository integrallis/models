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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * #425: a session prefix carries no native attention state.
 *
 * <p>The batched attention kernel serves only a backend's default sequence; sessions and their
 * forks run on the Java KV cache. A portable session prefix must therefore neither export the
 * kernel's retained store (it belongs to the default sequence, not to the prefix) nor overwrite a
 * receiver's store on import. This kernel retains state and offers to export it, so a backend that
 * still bundled it is caught without the native library.
 */
@Tag("unit")
class PortablePrefixAttentionOwnershipTest {

  /** Never attends; holds retained state and records every export and import. */
  static final class RetainingKernel implements BatchedCausalAttentionKernel {
    int writes;
    int reads;

    @Override
    public boolean isEligible(
        int layer,
        int startPosition,
        int batchSize,
        int numHeads,
        int numKvHeads,
        int keyLength,
        int valueLength,
        int maxSequenceLength,
        int slidingWindow) {
      return false;
    }

    @Override
    public void attend(
        float[] output,
        float[] query,
        float[] key,
        float[] value,
        int layer,
        int startPosition,
        int batchSize,
        int numHeads,
        int numKvHeads,
        int keyLength,
        int valueLength,
        int maxSequenceLength,
        int slidingWindow) {
      throw new AssertionError("a session path must not reach the native kernel");
    }

    @Override
    public boolean supportsPortableState() {
      return true;
    }

    @Override
    public long retainedStateBytes() {
      return 8;
    }

    @Override
    public void writeRetainedState(java.io.DataOutput output) throws IOException {
      writes++;
      output.writeLong(0x5a5a5a5a5a5a5a5aL);
    }

    @Override
    public void readRetainedState(java.io.DataInput input) throws IOException {
      reads++;
      input.readLong();
    }
  }

  private static final int[] PROMPT = {2, 3, 4, 5, 6};

  private static Path model(Path dir) throws IOException {
    return PureJavaBackendTest.buildNanoModelFile(dir, new Random(42), GgufTensorType.F32, 2, 2);
  }

  private static byte[] export(PureJavaBackend donor) throws IOException {
    var session = donor.openSession();
    donor.prefill(session, PROMPT, 0);
    var buffer = new ByteArrayOutputStream();
    donor.writePrefix(donor.freezePrefix(session), PROMPT.length, new DataOutputStream(buffer));
    return buffer.toByteArray();
  }

  @Test
  void aSessionPrefixExportsNoAttentionStateAndContinuesExactly(@TempDir Path dir)
      throws Exception {
    Path model = model(dir);
    RetainingKernel donorKernel = new RetainingKernel();
    RetainingKernel receiverKernel = new RetainingKernel();
    try (var donor = PureJavaBackend.load(model, GgufBatchedMatrixKernel.none(), donorKernel);
        var receiver =
            PureJavaBackend.load(model, GgufBatchedMatrixKernel.none(), receiverKernel)) {
      assertThat(donor.supportsPortablePrefixes()).isTrue();
      var session = donor.openSession();
      donor.prefill(session, PROMPT, 0);
      var prefix = donor.freezePrefix(session);
      var buffer = new ByteArrayOutputStream();
      donor.writePrefix(prefix, PROMPT.length, new DataOutputStream(buffer));
      byte[] bytes = buffer.toByteArray();
      assertThat(donorKernel.writes)
          .as("default-sequence state must not ride on a session prefix")
          .isZero();
      assertThat(ByteBuffer.wrap(bytes, 32, 4).getInt()).as("attention state length").isZero();
      assertThat((long) bytes.length).isEqualTo(donor.portablePrefixBytes(PROMPT.length));

      var imported =
          receiver.readPrefix(
              new DataInputStream(new ByteArrayInputStream(bytes)), PROMPT.length, bytes.length);
      assertThat(receiverKernel.reads)
          .as("an import must not touch the receiver's kernel")
          .isZero();

      // The donor's fork, the receiver's fork of the import, and the receiver recomputing it.
      try (var fromDonor = donor.fork(prefix);
          var fromImport = receiver.fork(imported);
          var recomputed = receiver.openSession()) {
        receiver.prefill(recomputed, PROMPT, 0);
        float[] expected = donor.forward(fromDonor, 9, PROMPT.length);
        assertThat(receiver.forward(fromImport, 9, PROMPT.length)).containsExactly(expected);
        assertThat(receiver.forward(recomputed, 9, PROMPT.length)).containsExactly(expected);
      }
    }
  }

  @Test
  void anImportCarryingAttentionStateIsRefusedAndLeavesTheReceiverUsable(@TempDir Path dir)
      throws Exception {
    Path model = model(dir);
    RetainingKernel receiverKernel = new RetainingKernel();
    try (var donor =
            PureJavaBackend.load(model, GgufBatchedMatrixKernel.none(), new RetainingKernel());
        var receiver =
            PureJavaBackend.load(model, GgufBatchedMatrixKernel.none(), receiverKernel)) {
      byte[] honest = export(donor);
      // A peer that bundles eight bytes of attention state after the identity.
      byte[] forged = new byte[honest.length + 8];
      System.arraycopy(honest, 0, forged, 0, 32);
      ByteBuffer.wrap(forged, 32, 4).putInt(8);
      System.arraycopy(honest, 36, forged, 44, honest.length - 36);
      assertThatThrownBy(
              () ->
                  receiver.readPrefix(
                      new DataInputStream(new ByteArrayInputStream(forged)),
                      PROMPT.length,
                      forged.length))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("attention state");
      assertThat(receiverKernel.reads).isZero();
      assertThat(
              receiver
                  .readPrefix(
                      new DataInputStream(new ByteArrayInputStream(honest)),
                      PROMPT.length,
                      honest.length)
                  .checkpoint())
          .isEqualTo(PROMPT.length);
    }
  }
}
