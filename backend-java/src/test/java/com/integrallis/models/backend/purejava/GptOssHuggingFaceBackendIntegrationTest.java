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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.runtime.chat.ToolCallScanner;
import com.integrallis.models.runtime.chat.ToolSyntax;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Real-checkpoint compatibility gate for the Java-native GPT-OSS MXFP4 path. */
@Tag("integration")
class GptOssHuggingFaceBackendIntegrationTest {

  private static final String REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee";
  private static final Map<String, String> PINNED_FILES =
      Map.of(
          "config.json", "3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce",
          "tokenizer.json", "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
          "tokenizer_config.json",
              "9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070",
          "model.safetensors.index.json",
              "0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1",
          "model-00000-of-00002.safetensors",
              "16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427",
          "model-00001-of-00002.safetensors",
              "4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea",
          "model-00002-of-00002.safetensors",
              "a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303");
  private static final int[] PROMPT_TOKENS = {
    200006, 17360, 200008, 3575, 553, 17554, 162016, 11, 261, 4410, 6439, 2359, 22203, 656, 7788,
    17527, 558, 87447, 100594, 25, 220, 1323, 19, 12, 3218, 279, 30377, 289, 25, 14093, 279, 2,
    13888, 18403, 25, 8450, 11, 49159, 11, 1721, 13, 21030, 2804, 413, 7360, 395, 1753, 3176, 13,
    200007, 200006, 1428, 200008, 864, 1001, 162108, 6439, 13, 200007, 200006, 173781
  };
  private static final int[] ORACLE_GENERATED_TOKENS = {
    200005, 35644, 200008, 976, 1825, 31064, 25, 392, 864, 1001, 162108, 6439, 3692, 3164, 1682,
    261, 4590, 4994, 13, 30456, 6052, 25, 13114, 11, 172308, 11, 129636, 11, 16453, 62360, 11, 363,
    72013, 627, 11, 5178, 13, 6214, 1308, 1001, 13, 63659, 63122, 25, 13114, 13, 2604, 172308, 13,
    17291, 6052, 25, 13114, 13, 200007, 200006, 173781, 200005, 17196, 200008, 21220, 13, 200002
  };

  @Test
  void executesThePinnedOfficial20BCheckpoint() {
    Path directory = fixtureDirectory();
    assertPinnedCheckpoint(directory);
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "128");

    try (PureJavaBackend backend = PureJavaBackend.load(directory)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("gpt-oss");
      assertThat(backend.metadata().vocabSize()).isEqualTo(201_088);
      assertThat(backend.contextCapacity()).isEqualTo(128);
      ModelPrompt rendered =
          ChatTemplate.GPT_OSS.render(List.of(ChatMessage.user("Name one JVM language.")));
      int[] prompt = backend.tokenizer().encode(rendered);
      assertThat(prompt).as("official tokenizer at revision %s", REVISION).isEqualTo(PROMPT_TOKENS);

      long started = System.nanoTime();
      float[] logits = backend.prefill(prompt, 0);
      double elapsedSeconds = (System.nanoTime() - started) / 1_000_000_000.0;

      assertThat(logits).hasSize(201_088);
      assertThat(allFinite(logits)).isTrue();
      int[] top = topTokenIds(logits, 10);
      assertThat(top[0]).isEqualTo(200_005);
      assertThat(backend.tokenizer().decode(top[0])).isEqualTo("<|channel|>");
      System.out.printf(
          "GPT_OSS_20B promptTokens=%d elapsedSeconds=%.3f argmax=%d decoded=%s top=%s%n",
          prompt.length,
          elapsedSeconds,
          top[0],
          backend.tokenizer().decode(top[0]),
          topLogits(logits, top));
      compareOracleWhenConfigured(logits, top);
      int[] generated = greedyContinuation(backend, prompt.length, top[0]);
      String decoded = backend.tokenizer().decode(generated);
      System.out.printf(
          "GPT_OSS_20B_GENERATION tokens=%s decoded=%s%n", Arrays.toString(generated), decoded);
      assertThat(generated).startsWith(Arrays.copyOf(ORACLE_GENERATED_TOKENS, 46));
      assertThat(generated[generated.length - 1]).isEqualTo(backend.tokenizer().eosToken());
      ToolCallScanner.Result response = ToolCallScanner.scan(decoded, ToolSyntax.HARMONY);
      assertThat(response.content()).isEqualTo("Java.");
      assertThat(response.toolCalls()).isEmpty();
    } finally {
      if (previous == null) {
        System.clearProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
      } else {
        System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
      }
    }
  }

  private static void compareOracleWhenConfigured(float[] actual, int[] actualTop) {
    String configured = System.getProperty("models.fixtures.gptOssOracleLogits", "");
    if (configured.isBlank()) {
      return;
    }
    Path path = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isRegularFile(path), "GPT-OSS oracle logits are not installed");
    float[] expected;
    try {
      byte[] encoded = Files.readAllBytes(path);
      assertThat(encoded).hasSize(Math.multiplyExact(actual.length, Float.BYTES));
      expected = new float[actual.length];
      ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(expected);
    } catch (IOException failure) {
      throw new AssertionError("cannot read GPT-OSS oracle logits", failure);
    }

    double dot = 0.0;
    double actualSquared = 0.0;
    double expectedSquared = 0.0;
    double squaredError = 0.0;
    double maxAbsoluteError = 0.0;
    for (int index = 0; index < actual.length; index++) {
      dot += (double) actual[index] * expected[index];
      actualSquared += (double) actual[index] * actual[index];
      expectedSquared += (double) expected[index] * expected[index];
      double error = actual[index] - expected[index];
      squaredError += error * error;
      maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(error));
    }
    int[] expectedTop = topTokenIds(expected, actualTop.length);
    long topOverlap =
        Arrays.stream(actualTop)
            .filter(
                actualToken -> Arrays.stream(expectedTop).anyMatch(value -> value == actualToken))
            .count();
    double cosine = dot / Math.sqrt(actualSquared * expectedSquared);
    double rootMeanSquaredError = Math.sqrt(squaredError / actual.length);
    assertThat(actualTop[0]).isEqualTo(expectedTop[0]);
    assertThat(topOverlap).isEqualTo(10);
    assertThat(cosine).isGreaterThan(0.9975);
    assertThat(rootMeanSquaredError).isLessThan(0.30);
    assertThat(maxAbsoluteError).isLessThan(1.60);
    System.out.printf(
        "GPT_OSS_20B_ORACLE cosine=%.9f rmse=%.9f maxAbs=%.9f top10Overlap=%d "
            + "expectedTop=%s%n",
        cosine,
        rootMeanSquaredError,
        maxAbsoluteError,
        topOverlap,
        topLogits(expected, expectedTop));
  }

  private static Path fixtureDirectory() {
    String configured = System.getProperty("models.fixtures.gptOssHuggingFaceDirectory", "");
    assumeTrue(!configured.isBlank(), "set models.fixtures.gptOssHuggingFaceDirectory");
    Path directory = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isDirectory(directory), "GPT-OSS Hugging Face fixture is not installed");
    return directory;
  }

  private static int[] greedyContinuation(
      PureJavaBackend backend, int promptTokens, int firstToken) {
    int[] generated = new int[ORACLE_GENERATED_TOKENS.length];
    generated[0] = firstToken;
    int count = 1;
    while (!backend.tokenizer().isEndOfGeneration(generated[count - 1])
        && count < generated.length) {
      float[] logits = backend.forward(generated[count - 1], promptTokens + count - 1);
      generated[count] = topTokenIds(logits, 1)[0];
      count++;
    }
    return Arrays.copyOf(generated, count);
  }

  private static void assertPinnedCheckpoint(Path directory) {
    PINNED_FILES.forEach(
        (name, expected) -> {
          Path file = directory.resolve(name);
          assertThat(file).as("GPT-OSS 20B file at revision %s", REVISION).isRegularFile();
          assertThat(sha256(file)).as(name).isEqualTo(expected);
        });
  }

  private static String sha256(Path path) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      try (InputStream input = Files.newInputStream(path)) {
        byte[] buffer = new byte[1024 * 1024];
        int read;
        while ((read = input.read(buffer)) >= 0) {
          digest.update(buffer, 0, read);
        }
      }
      return HexFormat.of().formatHex(digest.digest());
    } catch (IOException failure) {
      throw new AssertionError("cannot hash " + path, failure);
    } catch (NoSuchAlgorithmException failure) {
      throw new IllegalStateException("SHA-256 is unavailable", failure);
    }
  }

  private static int[] topTokenIds(float[] logits, int count) {
    return IntStream.range(0, logits.length)
        .boxed()
        .sorted(Comparator.<Integer>comparingDouble(index -> logits[index]).reversed())
        .limit(count)
        .mapToInt(Integer::intValue)
        .toArray();
  }

  private static boolean allFinite(float[] values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        return false;
      }
    }
    return true;
  }

  private static String topLogits(float[] logits, int[] tokenIds) {
    return Arrays.stream(tokenIds)
        .mapToObj(token -> token + ":" + logits[token])
        .toList()
        .toString();
  }
}
