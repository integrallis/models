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
 * Micro-benchmark measuring Unigram tokenization throughput before/after IntArrayBuilder fix for
 * #435.
 */
class GgufTokenizerUnigramThroughputTest {

  private static final String[] PROMPTS = {
    "The quick brown fox jumps over the lazy dog.",
    "In 2024, the company announced a new partnership with technology leaders.",
    "Machine learning models require careful tuning to achieve optimal performance metrics.",
    "The Java Virtual Machine provides a robust platform for cross-platform development.",
    "Tokenization is the process of converting text into individual tokens for model input.",
    "Open-source software has transformed the way developers collaborate on projects.",
    "Neural networks with billions of parameters can exhibit emergent capabilities.",
    "The future of artificial intelligence depends on responsible research and deployment.",
  };

  private static GgufMetadata createUnigramMetadata() {
    List<String> tokens = new ArrayList<>();
    tokens.add("<pad>");
    tokens.add("<unk>");
    tokens.add("<s>");
    tokens.add("</s>");

    // Vocabulary entries for T5-style unigram
    String[] vocab = {
      "the",
      "quick",
      "brown",
      "fox",
      "jumps",
      "over",
      "lazy",
      "dog",
      "in",
      "company",
      "announced",
      "new",
      "partnership",
      "with",
      "technology",
      "leaders",
      "machine",
      "learning",
      "models",
      "require",
      "careful",
      "tuning",
      "to",
      "achieve",
      "optimal",
      "performance",
      "metrics",
      "java",
      "virtual",
      "machine",
      "provides",
      "robust",
      "platform",
      "for",
      "cross",
      "platform",
      "development",
      "tokenization",
      "is",
      "process",
      "of",
      "converting",
      "text",
      "into",
      "individual",
      "tokens",
      "model",
      "input",
      "open",
      "source",
      "software",
      "has",
      "transformed",
      "way",
      "developers",
      "collaborate",
      "on",
      "projects",
      "neural",
      "networks",
      "with",
      "billions",
      "parameters",
      "can",
      "exhibit",
      "emergent",
      "capabilities",
      "future",
      "artificial",
      "intelligence",
      "depends",
      "responsible",
      "research",
      "deployment",
      // Single characters and subwords
      "a",
      "b",
      "c",
      "d",
      "e",
      "f",
      "g",
      "h",
      "i",
      "j",
      "k",
      "l",
      "m",
      "n",
      "o",
      "p",
      "q",
      "r",
      "s",
      "t",
      "u",
      "v",
      "w",
      "x",
      "y",
      "z",
      "T",
      "I",
      "J",
      "V",
      "M",
      "A",
      "N",
      "▁the",
      "▁quick",
      "▁brown",
      "▁fox",
      "▁jumps",
      "▁over",
      "▁lazy",
      "▁dog",
      "▁in",
      "▁company",
      "▁machine",
      "▁learning",
      "▁models",
      "▁java",
      "▁virtual",
      "▁machine",
      "▁provides",
      "▁robust",
      "▁tokenization",
      "▁is",
      "▁process",
      "▁of",
      "▁converting",
      "▁open",
      "▁source",
      "▁software",
      "▁has",
      "▁transformed",
      "▁neural",
      "▁networks",
      "▁with",
      "▁billions",
      "▁parameters",
      "▁future",
      "▁artificial",
      "▁intelligence",
      "▁depends",
      "ing",
      "ed",
      "er",
      "est",
      "ly",
      "tion",
      "ment",
      "ness",
    };
    for (String v : vocab) {
      tokens.add(v);
    }

    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    entries.put(
        "tokenizer.ggml.tokens",
        new GgufMetadataValue.ArrayValue(
            GgufValueType.STRING,
            tokens.stream()
                .map(token -> (GgufMetadataValue) new GgufMetadataValue.StringValue(token))
                .toList()));
    entries.put("tokenizer.ggml.model", new GgufMetadataValue.StringValue("t5"));
    entries.put("tokenizer.ggml.unknown_token_id", new GgufMetadataValue.Uint32Value(1));
    entries.put("tokenizer.ggml.bos_token_id", new GgufMetadataValue.Uint32Value(2));
    entries.put("tokenizer.ggml.eos_token_id", new GgufMetadataValue.Uint32Value(3));
    entries.put("tokenizer.ggml.add_bos_token", new GgufMetadataValue.Uint32Value(0));
    entries.put("tokenizer.ggml.add_eos_token", new GgufMetadataValue.Uint32Value(0));
    entries.put("tokenizer.ggml.token_type_count", new GgufMetadataValue.Uint32Value(1));
    return new GgufMetadata(entries);
  }

  @Test
  void measureUnigramTokenizationThroughput() {
    var tokenizer = GgufTokenizer.fromMetadata(createUnigramMetadata());

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

    System.out.printf("%n=== GgufTokenizer Unigram Throughput ===%n");
    System.out.printf("Vocab size:  %,d%n", tokenizer.vocabSize());
    System.out.printf("Iterations:  %,d prompts x %d repetitions%n", PROMPTS.length, iterations);
    System.out.printf("Total tokens encoded: %,d%n", totalTokens);
    System.out.printf("Elapsed:     %.1f ms%n", elapsedMs);
    System.out.printf("Throughput:  %.1f tokens/sec%n", tokensPerSec);
    System.out.printf("Throughput:  %.1f prompts/sec%n", promptsPerSec);
  }
}
