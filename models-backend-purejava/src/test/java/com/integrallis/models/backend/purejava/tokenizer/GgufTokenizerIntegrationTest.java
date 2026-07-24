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
package com.integrallis.models.backend.purejava.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assumptions.assumeThat;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Integration tests for the BPE tokenizer loaded from a real GGUF model. Validates encoding and
 * decoding against the Qwen3-0.6B vocabulary.
 */
@Tag("integration")
class GgufTokenizerIntegrationTest {

  private static final Path MODEL_PATH =
      Path.of(System.getProperty("user.home"), ".jvllm", "models", "Qwen3-0.6B-Q4_0.gguf");

  private static GgufTokenizer tokenizer;

  @BeforeAll
  static void loadTokenizer() throws IOException {
    assumeThat(Files.exists(MODEL_PATH))
        .as("Qwen3-0.6B-Q4_0.gguf must be present at %s", MODEL_PATH)
        .isTrue();

    Arena arena = Arena.ofShared();
    GgufFile file = GgufParser.parse(MODEL_PATH, arena);
    tokenizer = GgufTokenizer.fromMetadata(file.metadata());
  }

  @Nested
  class VocabularyProperties {

    @Test
    void vocabSizeIsReasonable() {
      // Qwen3 uses a large vocabulary (151k+ tokens)
      assertThat(tokenizer.vocabSize()).isGreaterThan(100_000);
    }

    @Test
    void bosTokenIsValid() {
      int bos = tokenizer.bosToken();
      assertThat(bos).isGreaterThanOrEqualTo(0);
      assertThat(bos).isLessThan(tokenizer.vocabSize());
    }

    @Test
    void eosTokenIsValid() {
      int eos = tokenizer.eosToken();
      assertThat(eos).isGreaterThanOrEqualTo(0);
      assertThat(eos).isLessThan(tokenizer.vocabSize());
    }

    @Test
    void bosAndEosAreDifferent() {
      assertThat(tokenizer.bosToken()).isNotEqualTo(tokenizer.eosToken());
    }
  }

  @Nested
  class Encoding {

    @Test
    void encodesHelloWorldToNonEmpty() {
      int[] tokens = tokenizer.encode("Hello, world!");
      assertThat(tokens).isNotEmpty();
      assertThat(tokens.length).isGreaterThanOrEqualTo(1);
      assertThat(tokens.length).isLessThan(20); // shouldn't be excessively fragmented
    }

    @Test
    void encodesEmptyStringToEmpty() {
      assertThat(tokenizer.encode("")).isEmpty();
    }

    @Test
    void allTokenIdsAreWithinVocab() {
      int[] tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog.");
      for (int token : tokens) {
        assertThat(token)
            .as("Token ID must be within vocabulary range")
            .isGreaterThanOrEqualTo(0)
            .isLessThan(tokenizer.vocabSize());
      }
    }

    @Test
    void longerTextProducesMoreTokens() {
      int[] short_ = tokenizer.encode("Hi");
      int[] long_ = tokenizer.encode("Hello, this is a longer sentence with many words.");
      assertThat(long_.length).isGreaterThan(short_.length);
    }

    @Test
    void encodesUnicodeText() {
      int[] tokens = tokenizer.encode("日本語テスト 🎉");
      assertThat(tokens).isNotEmpty();
      for (int token : tokens) {
        assertThat(token).isGreaterThanOrEqualTo(0).isLessThan(tokenizer.vocabSize());
      }
    }

    @Test
    void encodesCodeSnippet() {
      int[] tokens = tokenizer.encode("def fibonacci(n):\n    if n <= 1: return n\n");
      assertThat(tokens).isNotEmpty();
      assertThat(tokens.length).isGreaterThan(5);
    }

    @Test
    void encodesChatMlControlTokensLikeLlamaCpp() {
      int[] tokens = tokenizer.encode("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n");

      assertThat(tokens).containsExactly(151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198);
      assertThat(tokenizer.isEndOfGeneration(151645)).isTrue();
    }
  }

  @Nested
  class Decoding {

    @Test
    void decodesSingleTokenToNonEmpty() {
      // Decode a mid-range token (likely a common word or subword)
      String decoded = tokenizer.decode(100);
      assertThat(decoded).isNotNull();
    }

    @Test
    void decodeArrayProducesReadableText() {
      int[] tokens = tokenizer.encode("Hello world");
      String decoded = tokenizer.decode(tokens);
      // Decoded text should contain the key content (may not be exact due to BPE)
      assertThat(decoded).isNotEmpty();
    }

    @Test
    void roundTripForSimpleText() {
      String original = "Hello";
      int[] encoded = tokenizer.encode(original);
      String decoded = tokenizer.decode(encoded);
      // BPE tokenizers should preserve the original text through encode/decode
      assertThat(decoded).isEqualTo(original);
    }

    @Test
    void roundTripForSentence() {
      String original = "The quick brown fox jumps over the lazy dog.";
      int[] encoded = tokenizer.encode(original);
      String decoded = tokenizer.decode(encoded);
      assertThat(decoded).isEqualTo(original);
    }

    @Test
    void roundTripForMultiline() {
      String original = "Line 1\nLine 2\nLine 3";
      int[] encoded = tokenizer.encode(original);
      String decoded = tokenizer.decode(encoded);
      assertThat(decoded).isEqualTo(original);
    }
  }
}
