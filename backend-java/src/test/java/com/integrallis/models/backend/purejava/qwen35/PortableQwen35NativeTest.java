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
package com.integrallis.models.backend.purejava.qwen35;

import static org.assertj.core.api.Assertions.*;

import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import java.io.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.HexFormat;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Runs against the host's built native library, including native recurrent/activation kernels. */
@Tag("integration")
class PortableQwen35NativeTest {
  @Test
  void nativeHybridCheckpointsPreserveCompleteContinuation(@TempDir Path dir) throws Exception {
    exercise(Qwen35ForwardPassTest.writeToyModel(dir, false, 64), new int[] {2, 3, 4, 5, 6});
  }

  @Test
  @Tag("model-fixture")
  void pinnedQwen35FourBContinuesExactlyThroughTheNativeWrapper() throws Exception {
    var file = Path.of(System.getProperty("models.qwen35.portable.model"));
    var digest = MessageDigest.getInstance("SHA-256");
    try (var input = new java.security.DigestInputStream(Files.newInputStream(file), digest)) {
      input.transferTo(OutputStream.nullOutputStream());
    }
    assertThat(HexFormat.of().formatHex(digest.digest()))
        .isEqualTo("00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4");
    int[] tokens = new int[33];
    for (int index = 0; index < tokens.length; index++) tokens[index] = 500 + index;
    exercise(file, tokens);
  }

  private static void exercise(Path file, int[] tokens) throws Exception {
    var library = Path.of(System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY));
    try (var donor = RustFfmBackend.load(file, library);
        var receiver = RustFfmBackend.load(file, library);
        var source = donor.openSession();
        var control = receiver.openSession()) {
      assertThat(donor.supportsPortablePrefixes()).isTrue();
      donor.prefill(source, tokens, 0);
      receiver.prefill(control, tokens, 0);
      var prefix = donor.freezePrefix(source);
      var bytes = new ByteArrayOutputStream();
      donor.writePrefix(prefix, tokens.length, new DataOutputStream(bytes));
      assertThat(bytes.size()).isEqualTo(donor.portablePrefixBytes(tokens.length));
      var imported =
          receiver.readPrefix(
              new DataInputStream(new ByteArrayInputStream(bytes.toByteArray())),
              tokens.length,
              bytes.size());
      receiver.validatePrefixTokens(imported, tokens);
      assertThatThrownBy(() -> receiver.validatePrefixTokens(imported, new int[tokens.length]))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("history");
      try (var branch = receiver.fork(imported);
          var sibling = receiver.fork(imported)) {
        assertThat(receiver.sharesPrefixStorage(branch, sibling)).isTrue();
        for (int step = 0; step < 4; step++) {
          int token = 2 + step;
          assertThat(receiver.forward(branch, token, tokens.length + step))
              .containsExactly(receiver.forward(control, token, tokens.length + step));
        }
        receiver.reset(branch);
        assertThat(branch.checkpoint()).isEqualTo(tokens.length);
        assertThat(receiver.forward(branch, 7, tokens.length))
            .containsExactly(receiver.forward(sibling, 7, tokens.length));
        assertThat(receiver.sharesPrefixStorage(branch, sibling)).isTrue();
        receiver.reset(branch);
        receiver.rewind(control, tokens.length);
        assertThat(receiver.prefill(branch, new int[] {3, 4, 5}, tokens.length))
            .containsExactly(receiver.prefill(control, new int[] {3, 4, 5}, tokens.length));
      }
    }
  }
}
