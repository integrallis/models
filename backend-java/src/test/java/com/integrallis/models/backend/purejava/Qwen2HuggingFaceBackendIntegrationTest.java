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
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class Qwen2HuggingFaceBackendIntegrationTest {

  private static final String MODEL_SHA256 =
      "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe";
  private static final int[] CHATML_TOKENS = {
    151644, 872, 198, 675, 825, 72379, 4128, 13, 151645, 198, 151644, 77091, 198
  };
  private static final float[] HUGGING_FACE_FIRST_LOGITS = {
    7.1711001f,
    10.372878f,
    6.2867732f,
    6.8912621f,
    4.0959649f,
    4.4131360f,
    7.8144817f,
    7.5291200f,
    7.0460711f,
    8.0585680f,
    5.2136788f,
    6.7079167f,
    8.3307886f,
    11.508839f,
    5.7097855f,
    6.6055269f
  };

  @Test
  void matchesPinnedHuggingFaceFloat32ReferenceLogits() throws Exception {
    Path directory = fixtureDirectory();
    assertThat(sha256(directory.resolve("model.safetensors"))).isEqualTo(MODEL_SHA256);

    try (PureJavaBackend backend = PureJavaBackend.load(directory)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("qwen2");
      assertThat(backend.metadata().vocabSize()).isEqualTo(151936);
      assertThat(
              backend
                  .tokenizer()
                  .encodeControl(
                      "<|im_start|>user\nName one JVM language.<|im_end|>\n"
                          + "<|im_start|>assistant\n"))
          .containsExactly(CHATML_TOKENS);

      float[] logits = backend.prefill(CHATML_TOKENS, 0);

      assertThat(logits).hasSize(151936);
      for (int index = 0; index < HUGGING_FACE_FIRST_LOGITS.length; index++) {
        assertThat(logits[index])
            .as("logit %s", index)
            .isCloseTo(HUGGING_FACE_FIRST_LOGITS[index], within(1.0e-3f));
      }
      assertThat(argmax(logits)).isEqualTo(3966);
      assertThat(logits[3966]).isCloseTo(20.877668f, within(1.0e-3f));
    }
  }

  private static Path fixtureDirectory() {
    String configured = System.getProperty("models.fixtures.qwen25HuggingFaceDirectory", "");
    assumeTrue(!configured.isBlank(), "set models.fixtures.qwen25HuggingFaceDirectory");
    Path directory = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isDirectory(directory), "Qwen 2.5 Hugging Face fixture is not installed");
    return directory;
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }

  private static String sha256(Path path) throws IOException, NoSuchAlgorithmException {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (var input = Files.newInputStream(path)) {
      byte[] buffer = new byte[1024 * 1024];
      int read;
      while ((read = input.read(buffer)) >= 0) {
        digest.update(buffer, 0, read);
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }
}
