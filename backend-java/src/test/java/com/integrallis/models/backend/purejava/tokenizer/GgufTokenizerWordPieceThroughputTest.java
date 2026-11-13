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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import com.integrallis.models.backend.purejava.gguf.GgufValueType;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Micro-benchmark measuring WordPiece tokenization throughput before/after trie optimization for
 * #437.
 *
 * <p>Not a correctness test -- it prints timing to stdout for before/after comparison.
 */
class GgufTokenizerWordPieceThroughputTest {

  private static final String[] PROMPTS = {
    "Transit-friendly APIs handle naive cafe riders.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require careful tuning to achieve optimal performance.",
    "Open-source software has transformed developer collaboration on complex projects.",
    "Neural networks with billions of parameters exhibit emergent capabilities.",
    "Tokenization converts text into individual tokens for model input.",
    "The Java Virtual Machine provides robust cross-platform development.",
    "Responsible AI research depends on careful deployment and monitoring.",
  };

  private static GgufMetadata createLargeWordPieceMetadata() {
    List<String> tokens = new ArrayList<>();
    tokens.add("[UNK]");
    tokens.add("[CLS]");
    tokens.add("[SEP]");

    // Alphabet and common subwords
    for (char c = 'a'; c <= 'z'; c++) {
      tokens.add(String.valueOf(c));
    }
    for (char c = 'A'; c <= 'Z'; c++) {
      tokens.add(String.valueOf(c));
    }
    for (char c = '0'; c <= '9'; c++) {
      tokens.add(String.valueOf(c));
    }

    // Common WordPiece prefixes (▁ = U+2581)
    String[] prefixes = {"the", "in", "to", "of", "and", "a", "is", "it", "for", "on"};
    for (String p : prefixes) {
      tokens.add("\u2581" + p);
    }

    // Multi-character subwords (vocabulary building blocks)
    String[] subwords = {
      "trans",
      "sit",
      "friend",
      "ly",
      "hand",
      "le",
      "naive",
      "ca",
      "fe",
      "rid",
      "ers",
      "ap",
      "is",
      "learn",
      "ing",
      "mod",
      "el",
      "token",
      "iza",
      "tion",
      "per",
      "form",
      "ance",
      "neur",
      "al",
      "net",
      "work",
      "para",
      "met",
      "er",
      "emerg",
      "ent",
      "capa",
      "bil",
      "ity",
      "soft",
      "ware",
      "devel",
      "op",
      "collab",
      "or",
      "ation",
      "complex",
      "pro",
      "ject",
      "cross",
      "plat",
      "form",
      "robust",
      "virtu",
      "al",
      "mach",
      "ine",
      "care",
      "ful",
      "tun",
      "achieve",
      "optim",
      "input",
      "convert",
      "individu",
      "al",
      "monit",
      "or",
      "deploy",
      "research",
      "depend",
      "respons",
      "ible",
      "trans",
      "form",
      "quick",
      "brown",
      "fox",
      "jump",
      "over",
      "lazy",
      "dog"
    };
    for (String sw : subwords) {
      tokens.add(sw);
    }

    // Add some longer combined tokens
    tokens.add("\u2581the");
    tokens.add("\u2581in");
    tokens.add("\u2581to");
    tokens.add("\u2581of");
    tokens.add("\u2581and");
    tokens.add("\u2581a");
    tokens.add("\u2581is");
    tokens.add("\u2581for");
    tokens.add("transit");
    tokens.add("friendly");
    tokens.add("handle");
    tokens.add("naive");
    tokens.add("cafe");
    tokens.add("riders");

    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put(
        "tokenizer.ggml.tokens",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.STRING,
            tokens.stream()
                .map(token -> (GgufMetadataValue) new GgufMetadataValue.StringValue(token))
                .toList()));
    entries.put("tokenizer.ggml.model", new GgufMetadataValue.StringValue("bert"));
    entries.put("tokenizer.ggml.unknown_token_id", new GgufMetadataValue.Uint32Value(0));
    entries.put("tokenizer.ggml.bos_token_id", new GgufMetadataValue.Uint32Value(1));
    entries.put("tokenizer.ggml.cls_token_id", new GgufMetadataValue.Uint32Value(1));
    entries.put("tokenizer.ggml.eos_token_id", new GgufMetadataValue.Uint32Value(2));
    entries.put("tokenizer.ggml.seperator_token_id", new GgufMetadataValue.Uint32Value(2));
    return new GgufMetadata(entries);
  }

  @Test
  void measureWordPieceTokenizationThroughput() {
    var tokenizer = GgufTokenizer.fromMetadata(createLargeWordPieceMetadata());

    // Warmup
    for (int i = 0; i < 1000; i++) {
      for (String prompt : PROMPTS) {
        tokenizer.encode(ModelPrompt.text(prompt));
      }
    }

    // Measurement
    int iterations = 5000;
    long totalTokens = 0;
    long start = System.nanoTime();
    for (int i = 0; i < iterations; i++) {
      for (String prompt : PROMPTS) {
        int[] tokens = tokenizer.encode(ModelPrompt.text(prompt));
        totalTokens += tokens.length;
      }
    }
    long elapsedNs = System.nanoTime() - start;
    double elapsedMs = elapsedNs / 1_000_000.0;
    double tokensPerSec = (totalTokens * 1_000_000_000.0) / elapsedNs;
    double promptsPerSec = (iterations * PROMPTS.length * 1_000_000_000.0) / elapsedNs;

    System.out.printf("%n=== GgufTokenizer WordPiece Throughput ===%n");
    System.out.printf("Vocab size:  %,d%n", tokenizer.vocabSize());
    System.out.printf("Iterations:  %,d prompts x %d repetitions%n", PROMPTS.length, iterations);
    System.out.printf("Total tokens encoded: %,d%n", totalTokens);
    System.out.printf("Elapsed:     %.1f ms%n", elapsedMs);
    System.out.printf("Throughput:  %.1f tokens/sec%n", tokensPerSec);
    System.out.printf("Throughput:  %.1f prompts/sec%n", promptsPerSec);
  }
}
