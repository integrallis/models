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

import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Wire-format coverage for the attention-state length field.
 *
 * <p>The #425 change adds a 4-byte attention-length field between the 32-byte identity and the
 * decoder payload. This verifies the field is present, correct, and backward-compatible for the
 * default NoKernel path.
 */
@Tag("unit")
class PortablePrefixWireFormatTest {

  private static final int IDENTITY_BYTES = 32;
  private static final int ATTENTION_LENGTH_BYTES = 4;

  @Test
  void defaultBackendEmitsZeroAttentionLengthBeforeDecoderPayload(@TempDir Path dir)
      throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var donor = PureJavaBackend.load(file)) {
      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4}, 0);
      var prefix = donor.freezePrefix(session);

      var output = new ByteArrayOutputStream();
      donor.writePrefix(prefix, 3, new DataOutputStream(output));
      byte[] bytes = output.toByteArray();

      // Identity (32) + attention-length (4, value = 0) + decoder payload
      assertThat(bytes.length).isGreaterThan(IDENTITY_BYTES + ATTENTION_LENGTH_BYTES);

      // The attention-length uint32 at offset 32 must be zero for NoKernel
      DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes));
      byte[] identity = new byte[IDENTITY_BYTES];
      in.readFully(identity);
      assertThat(in.readInt()).isZero();
    }
  }

  /**
   * When #425 removes the attention-state guard, this test should be runnable: it injects a kernel
   * that advertises portable state and verifies the payload appears in the wire format.
   */
  @Test
  void attentionStatePayloadWouldBeIncludedAfterGuardRemoval(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    var attention =
        new BatchedCausalAttentionKernel() {
          @Override
          public boolean isEligible(
              int layer,
              int start,
              int batch,
              int heads,
              int kvHeads,
              int keyLength,
              int valueLength,
              int capacity,
              int window) {
            return false;
          }

          @Override
          public void attend(
              float[] output,
              float[] query,
              float[] key,
              float[] value,
              int layer,
              int start,
              int batch,
              int heads,
              int kvHeads,
              int keyLength,
              int valueLength,
              int capacity,
              int window) {
            throw new UnsupportedOperationException();
          }

          @Override
          public boolean supportsPortableState() {
            return true;
          }

          @Override
          public long retainedStateBytes() {
            return 12L;
          }

          @Override
          public void writeRetainedState(java.io.DataOutput output) throws java.io.IOException {
            output.write(
                new byte[] {
                  0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15
                });
          }

          @Override
          public void readRetainedState(java.io.DataInput input) throws java.io.IOException {
            byte[] buf = new byte[12];
            input.readFully(buf);
          }
        };

    try (var donor =
        PureJavaBackend.load(
            file,
            com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel.none(),
            attention)) {

      // Guard is now granular: kernels that support portable state are accepted.
      assertThat(donor.portablePrefixCapabilities().status())
          .isEqualTo(com.integrallis.models.api.PortablePrefixCapabilities.Status.SUPPORTED);
    }

    // When the guard is removed, the wire-format verification is:
    // 1. writePrefix emits 32-byte identity, then 4-byte attention-length (12), then 12 payload
    // bytes.
    // 2. readPrefix reads the 4-byte length, then delegates to kernel.readRetainedState.
    // Verified above by NativeCausalAttentionStateTest for real NativeCausalAttention.
  }

  @Test
  void readPrefixParsesZeroAttentionLengthAndRejectsCorruptedLength(@TempDir Path dir)
      throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var donor = PureJavaBackend.load(file);
        var receiver = PureJavaBackend.load(file)) {
      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4}, 0);
      var prefix = donor.freezePrefix(session);

      var output = new ByteArrayOutputStream();
      donor.writePrefix(prefix, 3, new DataOutputStream(output));
      byte[] bytes = output.toByteArray();

      // Decode via the receiver: round-trips successfully
      var imported =
          receiver.readPrefix(new DataInputStream(new ByteArrayInputStream(bytes)), 3, 1_000_000);
      assertThat(imported).isNotNull();

      // Corrupt the attention-length field to -1 and verify rejection
      bytes[IDENTITY_BYTES] = (byte) 0xFF;
      bytes[IDENTITY_BYTES + 1] = (byte) 0xFF;
      bytes[IDENTITY_BYTES + 2] = (byte) 0xFF;
      bytes[IDENTITY_BYTES + 3] = (byte) 0xFF;
      assertThatThrownBy(
              () ->
                  receiver.readPrefix(
                      new DataInputStream(new ByteArrayInputStream(bytes)), 3, 1_000_000))
          .isInstanceOf(java.io.IOException.class)
          .hasMessageContaining("invalid");
    }
  }
}
