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

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.integrallis.models.api.*;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Native identity/framing contracts; real tensor continuation is covered by decoder tests and VPS
 * evidence.
 */
class PortableNativePrefixTest {
  private static final BackendDiagnostics DIAGNOSTICS =
      new BackendDiagnostics("rust-ffm", RustFfmBackend.PLAN_VERSION, Map.of(), List.of());

  private static PureJavaBackend decoder() throws IOException {
    var decoder = mock(PureJavaBackend.class);
    when(decoder.supportsSharedPrefixes()).thenReturn(true);
    when(decoder.supportsPortablePrefixes()).thenReturn(true);
    when(decoder.portablePrefixFingerprint()).thenReturn("a".repeat(64));
    return decoder;
  }

  @Test
  void actualKernelIdentityChangesWithExecutionSettingsAndRefusesVisionState() throws Exception {
    Path library = Path.of(System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY));
    String first;
    try (var kernel = RustGgufBatchedMatrixKernel.open(library, 1)) {
      assertThat(kernel.supportsPortablePrefixes()).isTrue();
      first = kernel.portablePrefixFingerprint();
      assertThat(first).matches("[0-9a-f]{64}");
    }
    try (var kernel = RustGgufBatchedMatrixKernel.open(library, 2)) {
      assertThat(kernel.portablePrefixFingerprint()).isNotEqualTo(first);
      if (kernel.configureImagePrefill()) {
        assertThat(kernel.supportsPortablePrefixes()).isFalse();
        assertThatThrownBy(kernel::portablePrefixFingerprint)
            .isInstanceOf(UnsupportedOperationException.class);
      }
    }
  }

  @Test
  void identifiedNativeEnvelopeRejectsWrongBinaryBeforeTouchingDecoder(@TempDir Path dir)
      throws Exception {
    Path binary = Files.writeString(dir.resolve("kernel.bin"), "identified-native-binary");
    Path sameBinary = Files.copy(binary, dir.resolve("another-location.bin"));
    Path wrongBinary = Files.writeString(dir.resolve("wrong.bin"), "different-native-binary");
    var donorDecoder = decoder();
    var receiverDecoder = decoder();
    var wrongDecoder = decoder();
    var prefix = mock(SharedInferencePrefix.class);
    when(donorDecoder.portablePrefixBytes(3)).thenReturn(64L);
    doAnswer(
            call -> {
              ((DataOutput) call.getArgument(2)).write(new byte[64]);
              return null;
            })
        .when(donorDecoder)
        .writePrefix(same(prefix), eq(3), any(DataOutput.class));
    when(receiverDecoder.readPrefix(any(DataInput.class), eq(3), eq(64L)))
        .thenAnswer(
            call -> {
              byte[] payload = new byte[64];
              ((DataInput) call.getArgument(0)).readFully(payload);
              assertThat(payload).containsOnly((byte) 0);
              return prefix;
            });
    try (var donor = new RustFfmBackend(donorDecoder, DIAGNOSTICS, binary);
        var receiver = new RustFfmBackend(receiverDecoder, DIAGNOSTICS, sameBinary);
        var wrong = new RustFfmBackend(wrongDecoder, DIAGNOSTICS, wrongBinary)) {
      assertThat(donor.supportsPortablePrefixes()).isTrue();
      assertThat(donor.portablePrefixFingerprint()).isEqualTo(receiver.portablePrefixFingerprint());
      assertThat(donor.portablePrefixFingerprint()).isNotEqualTo(wrong.portablePrefixFingerprint());
      verify(donorDecoder, times(1)).portablePrefixFingerprint();
      var output = new ByteArrayOutputStream();
      donor.writePrefix(prefix, 3, new DataOutputStream(output));
      assertThat((long) output.size()).isEqualTo(donor.portablePrefixBytes(3)).isEqualTo(96L);
      byte[] bytes = output.toByteArray();
      assertThat(receiver.readPrefix(input(bytes), 3, bytes.length)).isSameAs(prefix);
      assertThatThrownBy(() -> wrong.readPrefix(input(bytes), 3, bytes.length))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("identity");
      assertThatThrownBy(() -> receiver.readPrefix(input(bytes), 3, 83))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("budget");
      bytes[0] ^= 1;
      assertThatThrownBy(() -> receiver.readPrefix(input(bytes), 3, bytes.length))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("identity");
      verify(wrongDecoder, never()).readPrefix(any(), anyInt(), anyLong());
      verify(receiverDecoder, times(1)).readPrefix(any(), anyInt(), anyLong());
    }
  }

  @Test
  void unprovenNativeIdentityCannotAdvertisePortabilityButRetainsOwnedPrefixContract()
      throws Exception {
    var decoder = decoder();
    var session = mock(InferenceSession.class);
    var sibling = mock(InferenceSession.class);
    var prefix = mock(SharedInferencePrefix.class);
    var truncated = mock(SharedInferencePrefix.class);
    when(decoder.freezePrefix(session)).thenReturn(prefix);
    when(decoder.fork(prefix, SharedPrefixInferenceBackend.Branch.BASE)).thenReturn(sibling);
    when(decoder.sharesPrefixStorage(session, sibling)).thenReturn(true);
    when(decoder.truncatePrefix(prefix, 2)).thenReturn(truncated);
    when(decoder.supportsActivatedBranch()).thenReturn(true);
    try (var backend = new RustFfmBackend(decoder, DIAGNOSTICS)) {
      assertThat(backend.supportsPortablePrefixes()).isFalse();
      assertThatThrownBy(backend::portablePrefixFingerprint)
          .isInstanceOf(UnsupportedOperationException.class);
      assertThat(backend.supportsSharedPrefixes()).isTrue();
      assertThat(backend.supportsActivatedBranch()).isTrue();
      assertThat(backend.freezePrefix(session)).isSameAs(prefix);
      assertThat(backend.fork(prefix, SharedPrefixInferenceBackend.Branch.BASE)).isSameAs(sibling);
      assertThat(backend.sharesPrefixStorage(session, sibling)).isTrue();
      assertThat(backend.truncatePrefix(prefix, 2)).isSameAs(truncated);
    }
  }

  private static DataInputStream input(byte[] bytes) {
    return new DataInputStream(new ByteArrayInputStream(bytes));
  }
}
