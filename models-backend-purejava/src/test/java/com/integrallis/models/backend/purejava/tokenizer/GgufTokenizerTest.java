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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import com.integrallis.models.backend.purejava.gguf.GgufValueType;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GgufTokenizerTest {

  /**
   * Creates a small test vocabulary: 0: "h" 1: "e" 2: "l" 3: "o" 4: " " 5: "w" 6: "r" 7: "d" 8:
   * "he" 9: "ll" 10: "lo" 11: "hello" 12: "world" 13: "<0x41>" (byte A) 14: "<s>" (BOS) 15: "</s>"
   * (EOS)
   */
  private GgufMetadata createTestMetadata() {
    List<String> tokens =
        List.of(
            "h", "e", "l", "o", " ", "w", "r", "d", "he", "ll", "lo", "hello", "world", "<0x41>",
            "<s>", "</s>");
    List<Float> scores =
        List.of(
            -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -0.5f, -0.5f, -0.5f, 0.0f, 0.0f,
            -2.0f, 0.0f, 0.0f);
    // Merges: "h e" -> "he", "l l" -> "ll", "l o" -> "lo", "he ll" -> (not in vocab as hell),
    // "hel lo" -> (not direct)
    // Keep it simple: "h e" merges to "he" (rank 0), "l l" -> "ll" (rank 1), "l o" -> "lo" (rank
    // 2)
    List<String> merges = List.of("h e", "l l", "l o");

    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put(
        "tokenizer.ggml.tokens",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.STRING,
            tokens.stream()
                .map(s -> (GgufMetadataValue) new GgufMetadataValue.StringValue(s))
                .toList()));
    entries.put(
        "tokenizer.ggml.scores",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.FLOAT32,
            scores.stream()
                .map(f -> (GgufMetadataValue) new GgufMetadataValue.Float32Value(f))
                .toList()));
    entries.put(
        "tokenizer.ggml.merges",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.STRING,
            merges.stream()
                .map(s -> (GgufMetadataValue) new GgufMetadataValue.StringValue(s))
                .toList()));
    entries.put("tokenizer.ggml.bos_token_id", new GgufMetadataValue.Uint32Value(14));
    entries.put("tokenizer.ggml.eos_token_id", new GgufMetadataValue.Uint32Value(15));

    return new GgufMetadata(entries);
  }

  private GgufMetadata createByteLevelMetadata() {
    return createByteLevelMetadata(false, false);
  }

  private GgufMetadata createByteLevelMetadata(boolean addBosToken, boolean addEosToken) {
    List<String> tokens =
        List.of("<unk>", "h", "i", "\u0120", "hi", "hi\u0120", "<0x41>", "\u20ac", "<s>", "</s>");
    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put(
        "tokenizer.ggml.tokens",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.STRING,
            tokens.stream()
                .map(s -> (GgufMetadataValue) new GgufMetadataValue.StringValue(s))
                .toList()));
    entries.put(
        "tokenizer.ggml.merges",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.STRING,
            List.of("h i", "hi \u0120").stream()
                .map(s -> (GgufMetadataValue) new GgufMetadataValue.StringValue(s))
                .toList()));
    entries.put("tokenizer.ggml.model", new GgufMetadataValue.StringValue("gpt2"));
    entries.put("tokenizer.ggml.bos_token_id", new GgufMetadataValue.Uint32Value(8));
    entries.put("tokenizer.ggml.eos_token_id", new GgufMetadataValue.Uint32Value(9));
    entries.put("tokenizer.ggml.add_bos_token", new GgufMetadataValue.BoolValue(addBosToken));
    entries.put("tokenizer.ggml.add_eos_token", new GgufMetadataValue.BoolValue(addEosToken));
    return new GgufMetadata(entries);
  }

  private GgufMetadata createSentencePieceMetadata() {
    List<String> tokens =
        List.of(
            "<unk>",
            "<s>",
            "</s>",
            "\u2581",
            "a",
            "b",
            "c",
            "\u2581a",
            "ab",
            "bc",
            "\u2581ab",
            "<0x21>");
    List<Float> scores =
        List.of(0.0f, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f, -1.0f, 10.0f, 1.0f, 9.0f, 0.0f, -1.0f);

    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put(
        "tokenizer.ggml.tokens",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.STRING,
            tokens.stream()
                .map(s -> (GgufMetadataValue) new GgufMetadataValue.StringValue(s))
                .toList()));
    entries.put(
        "tokenizer.ggml.scores",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.FLOAT32,
            scores.stream()
                .map(f -> (GgufMetadataValue) new GgufMetadataValue.Float32Value(f))
                .toList()));
    entries.put("tokenizer.ggml.model", new GgufMetadataValue.StringValue("llama"));
    entries.put("tokenizer.ggml.bos_token_id", new GgufMetadataValue.Uint32Value(1));
    entries.put("tokenizer.ggml.eos_token_id", new GgufMetadataValue.Uint32Value(2));
    entries.put("tokenizer.ggml.add_bos_token", new GgufMetadataValue.BoolValue(true));
    entries.put("tokenizer.ggml.add_space_prefix", new GgufMetadataValue.BoolValue(true));
    return new GgufMetadata(entries);
  }

  @Nested
  class BasicEncoding {

    @Test
    void encodesSingleKnownToken() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createTestMetadata());

      // "hello" is token id 11
      int[] tokens = tokenizer.encode("hello");
      assertThat(tokens).contains(11);
    }

    @Test
    void encodesEmptyString() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createTestMetadata());
      assertThat(tokenizer.encode("")).isEmpty();
    }

    @Test
    void encodesNullReturnsEmpty() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createTestMetadata());
      assertThat(tokenizer.encode(null)).isEmpty();
    }
  }

  @Nested
  class Decoding {

    @Test
    void decodesKnownToken() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createTestMetadata());
      assertThat(tokenizer.decode(11)).isEqualTo("hello");
    }

    @Test
    void decodesByteToken() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createTestMetadata());
      // Token 13 is "<0x41>" which should decode to "A"
      assertThat(tokenizer.decode(13)).isEqualTo("A");
    }

    @Test
    void decodeArrayRoundTrips() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createTestMetadata());
      assertThat(tokenizer.decode(new int[] {11, 4, 12})).isEqualTo("hello world");
    }

    @Test
    void decodeOutOfRangeReturnsEmpty() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createTestMetadata());
      assertThat(tokenizer.decode(-1)).isEmpty();
      assertThat(tokenizer.decode(999)).isEmpty();
    }
  }

  @Nested
  class ByteLevelBpe {

    @Test
    void appliesRankedMergesAndRoundTripsSpace() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createByteLevelMetadata());

      int[] encoded = tokenizer.encode("hi ");

      assertThat(encoded).containsExactly(5);
      assertThat(tokenizer.decode(encoded)).isEqualTo("hi ");
      assertThat(tokenizer.decode(5)).isEqualTo("hi ");
    }

    @Test
    void fallsBackToExplicitByteToken() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createByteLevelMetadata());

      assertThat(tokenizer.encode("A")).containsExactly(6);
      assertThat(tokenizer.decode(new int[] {6})).isEqualTo("A");
    }

    @Test
    void skipsInvalidIdsAndEncodesUnknownVocabularyCharacters() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createByteLevelMetadata());

      assertThat(tokenizer.decode(new int[] {-1, 7, 99})).isEqualTo("\u20ac");
    }

    @Test
    void appliesConfiguredBosAndEosTokens() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createByteLevelMetadata(true, true));

      int[] encoded = tokenizer.encode("hi ");

      assertThat(encoded).containsExactly(8, 5, 9);
      assertThat(tokenizer.decode(encoded)).isEqualTo("hi ");
    }
  }

  @Nested
  class SentencePiece {

    @Test
    void appliesScoreOrderedMergesAndConfiguredBos() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createSentencePieceMetadata());

      int[] encoded = tokenizer.encode("abc");

      assertThat(encoded).containsExactly(1, 7, 9);
      assertThat(tokenizer.decode(encoded)).isEqualTo("abc");
    }

    @Test
    void addsBosForEmptyInput() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createSentencePieceMetadata());

      assertThat(tokenizer.encode("")).containsExactly(1);
    }

    @Test
    void roundTripsDummyPrefixAndByteFallback() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createSentencePieceMetadata());

      int[] encoded = tokenizer.encode("!");

      assertThat(encoded).containsExactly(1, 3, 11);
      assertThat(tokenizer.decode(encoded)).isEqualTo("!");
    }
  }

  @Nested
  class SpecialTokens {

    @Test
    void bosTokenId() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createTestMetadata());
      assertThat(tokenizer.bosToken()).isEqualTo(14);
    }

    @Test
    void eosTokenId() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createTestMetadata());
      assertThat(tokenizer.eosToken()).isEqualTo(15);
    }

    @Test
    void vocabSize() {
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(createTestMetadata());
      assertThat(tokenizer.vocabSize()).isEqualTo(16);
    }
  }

  @Nested
  class Errors {

    @Test
    void missingTokensThrows() {
      GgufMetadata empty = new GgufMetadata(Map.of());
      assertThatThrownBy(() -> GgufTokenizer.fromMetadata(empty))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("tokenizer.ggml.tokens");
    }
  }
}
