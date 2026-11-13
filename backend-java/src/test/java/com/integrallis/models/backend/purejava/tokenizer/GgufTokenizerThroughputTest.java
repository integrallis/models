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
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Micro-benchmark measuring tokenization throughput before/after allocator fixes for #435.
 *
 * <p>Not a correctness test -- it prints timing to stdout for before/after comparison. Run with
 * {@code --info} to see output.
 */
class GgufTokenizerThroughputTest {

  private static final int VOCAB_SIZE = 256;
  private static final String[] PROMPTS = {
    "The quick brown fox jumps over the lazy dog.",
    "In 2024, the company announced a new partnership with several technology leaders.",
    "Machine learning models require careful tuning to achieve optimal performance metrics.",
    "The Java Virtual Machine provides a robust platform for cross-platform development.",
    "Tokenization is the process of converting text into individual tokens for model input.",
    "Open-source software has transformed the way developers collaborate on complex projects.",
    "Neural networks with billions of parameters can exhibit emergent capabilities.",
    "The future of artificial intelligence depends on responsible research and deployment.",
  };

  static Path buildBpeTokenizerGguf(Path dir) throws Exception {
    List<String> tokens = new ArrayList<>();
    List<Float> scores = new ArrayList<>();
    for (int i = 0; i < VOCAB_SIZE; i++) {
      tokens.add("t" + i);
      scores.add(0.0f);
    }

    // Byte-level fallback tokens <0x00> through <0xFF>
    for (int b = 0; b < 256; b++) {
      tokens.set(b, String.format("<0x%02X>", b));
    }

    tokens.set(128, "Ġthe");
    tokens.set(129, "Ġquick");
    tokens.set(130, "Ġbrown");
    tokens.set(131, "Ġfox");
    tokens.set(132, "Ġjumps");
    tokens.set(133, "Ġover");
    tokens.set(134, "Ġlazy");
    tokens.set(135, "Ġdog");
    tokens.set(136, "ing");
    tokens.set(137, "tion");
    tokens.set(138, "ment");
    tokens.set(139, "Ġlearn");
    tokens.set(140, "Ġmodel");
    tokens.set(141, "Ġtoken");

    List<String> merges = new ArrayList<>();
    merges.add("Ġthe,Ġquick");
    merges.add("Ġbrown,Ġfox");
    merges.add("Ġjumps,Ġover");
    merges.add("Ġlazy,Ġdog");
    merges.add("learn,ing");
    merges.add("token,ize");
    merges.add("Ġmodel,s");

    var builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "llama")
            .addString("tokenizer.ggml.model", "gpt2")
            .addStringArray("tokenizer.ggml.tokens", tokens)
            .addFloat32Array("tokenizer.ggml.scores", scores)
            .addStringArray("tokenizer.ggml.merges", merges)
            .addUint32("tokenizer.ggml.bos_token_id", 0)
            .addUint32("tokenizer.ggml.eos_token_id", 1)
            .addUint32("tokenizer.ggml.add_bos_token", 0)
            .addUint32("tokenizer.ggml.add_eos_token", 0)
            .addUint32("tokenizer.ggml.token_type_count", 1);

    Path file = dir.resolve("tokenizer.gguf");
    Files.write(file, builder.build());
    return file;
  }

  @Test
  void measureBpeTokenizationThroughput(@TempDir Path dir) throws Exception {
    Path file = buildBpeTokenizerGguf(dir);

    try (Arena arena = Arena.ofConfined()) {
      var metadata = GgufParser.parse(file, arena).metadata();
      var tokenizer = GgufTokenizer.fromMetadata(metadata);

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

      System.out.printf("%n=== GgufTokenizer BPE Throughput ===%n");
      System.out.printf("Iterations:  %,d prompts x %d repetitions%n", PROMPTS.length, iterations);
      System.out.printf("Total tokens encoded: %,d%n", totalTokens);
      System.out.printf("Elapsed:     %.1f ms%n", elapsedMs);
      System.out.printf("Throughput:  %.1f tokens/sec%n", tokensPerSec);
      System.out.printf("Throughput:  %.1f prompts/sec%n", promptsPerSec);
    }
  }
}
