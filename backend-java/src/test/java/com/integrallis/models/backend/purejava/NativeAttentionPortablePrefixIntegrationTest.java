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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.nativekernel.NativeGgufMatrixKernelProvider;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.ServiceLoader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * #425 end-to-end: portable prefix round-trip with native attention active.
 *
 * <p>Prefills a session through the full model path with native attention injected via the {@link
 * NativeGgufMatrixKernelProvider} service path, exports the prefix, imports into an independently
 * loaded backend, and verifies exact continuation. This is the test that justifies removing the
 * ATTENTION_STATE_UNSUPPORTED guard.
 *
 * <p>Not tagged {@code slow} because the {@code assumeTrue} guards skip it immediately when the
 * native library is absent, keeping the default {@code test} task fast on machines without it.
 */
class NativeAttentionPortablePrefixIntegrationTest {

  /** Locate the native kernel provider through ServiceLoader if the library is present. */
  private static com.integrallis.models.backend.purejava.spi.GgufMatrixKernelProvider
      nativeAttentionProvider() {
    String configured = System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY);
    assumeTrue(configured != null, RustFfmBackend.LIBRARY_PATH_PROPERTY + " is not set");
    Path library = Path.of(configured);
    assumeTrue(Files.isRegularFile(library), library + " is not a file");

    for (com.integrallis.models.backend.purejava.spi.GgufMatrixKernelProvider provider :
        ServiceLoader.load(
            com.integrallis.models.backend.purejava.spi.GgufMatrixKernelProvider.class)) {
      if (provider instanceof NativeGgufMatrixKernelProvider) {
        return provider;
      }
    }
    assumeTrue(false, "NativeGgufMatrixKernelProvider not found via ServiceLoader");
    return null; // unreachable
  }

  @Test
  void nativeAttentionPrefixRoundTripsExactContinuation(@TempDir Path dir) throws Exception {
    var provider = nativeAttentionProvider();
    var matrix = provider.open(3).orElse(null);
    assumeTrue(matrix != null, "Native matrix kernel unavailable");
    var attention = provider.openAttention(3, matrix);
    assumeTrue(
        attention != BatchedCausalAttentionKernel.none(),
        "Native causal attention not supported by this library");

    // MHA model: heads == kv_heads, required by NativeCausalAttention eligibility.
    Path model =
        PureJavaBackendTest.buildNanoModelFile(
            dir,
            new Random(42),
            com.integrallis.models.backend.purejava.gguf.GgufTensorType.F32,
            2,
            2);

    var matrixRecv = provider.open(3).orElseThrow();
    var attentionRecv = provider.openAttention(3, matrixRecv);

    try (var donor = PureJavaBackend.load(model, matrix, attention);
        var receiver = PureJavaBackend.load(model, matrixRecv, attentionRecv)) {

      assertThat(donor.supportsPortablePrefixes()).isTrue();

      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4, 5}, 0);
      var prefix = donor.freezePrefix(session);
      assertThat(prefix.checkpoint()).isEqualTo(4);

      var output = new ByteArrayOutputStream();
      donor.writePrefix(prefix, 4, new DataOutputStream(output));
      byte[] bytes = output.toByteArray();

      var imported =
          receiver.readPrefix(
              new DataInputStream(new ByteArrayInputStream(bytes)), 4, bytes.length);
      assertThat(imported.checkpoint()).isEqualTo(4);

      try (var forkDonor = donor.fork(prefix);
          var forkRecv = receiver.fork(imported)) {
        float[] donorLogits = donor.forward(forkDonor, 9, 4);
        float[] recvLogits = receiver.forward(forkRecv, 9, 4);
        assertThat(recvLogits).containsExactly(donorLogits);
      }
    }
  }
}
