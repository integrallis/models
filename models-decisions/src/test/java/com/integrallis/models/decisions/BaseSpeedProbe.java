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
package com.integrallis.models.decisions;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * What a base costs on this workload, before any head is trained against it.
 *
 * <p>Swapping the base is the largest lever on latency and the most expensive one to try, because
 * a new base needs a fresh harvest and a fresh head. This measures the part that does not need
 * either: how long the document prefill and a question tail take. If the cheaper base is not
 * proportionally faster here it will not be faster once a head is fitted, and the harvest is not
 * worth running.
 *
 * <p>Reports both the one-off and the marginal cost, because the two scale differently with
 * document length and only the marginal one is paid per question.
 */
public final class BaseSpeedProbe {

  private static final int WARMUP = 2;
  private static final int REPEATS = 3;
  private static final int TAIL_COUNT = 10;

  private BaseSpeedProbe() {}

  /** Arguments: GGUF base, document file, one question. */
  public static void main(String[] args) throws IOException {
    Path model = Path.of(args[0]);
    String document = Files.readString(Path.of(args[1]), StandardCharsets.UTF_8).strip();
    String question = args.length > 2 ? args[2] : "What is the monthly fee?";

    PureJavaBackend backend;
    String kernel;
    try {
      backend = PureJavaBackend.load(model, RustGgufBatchedMatrixKernel.openBundled());
      kernel = "rust-ffm";
    } catch (RuntimeException | UnsatisfiedLinkError unavailable) {
      backend = PureJavaBackend.load(model);
      kernel = "pure-java";
    }

    try (PureJavaBackend held = backend) {
      SharedPrefixInferenceBackend sharing = held;
      int[] documentTokens = held.tokenizer().encode(document + "\n\nQuestion:");
      int[] tail =
          held.tokenizer()
              .encode(
                  " "
                      + question
                      + "\nIs the question answerable from the text above? Answer yes or no."
                      + "\nAnswer:");

      System.out.printf("%n  base %s%n", model.getFileName());
      System.out.printf("  kernel %s, hidden width %s%n", kernel, widthOf(held, sharing));
      System.out.printf("  document %d tokens, tail %d tokens%n%n", documentTokens.length, tail.length);

      double prefillBest = Double.MAX_VALUE;
      double tailBest = Double.MAX_VALUE;
      double sequentialBest = Double.MAX_VALUE;
      double batchedBest = Double.MAX_VALUE;
      // Warm-ups are discarded: the first passes pay class loading and page faults that a served
      // request would not, and reporting them would flatter whichever base ran second.
      for (int round = 0; round < WARMUP + REPEATS; round++) {
        long a = System.nanoTime();
        SharedInferencePrefix prefix;
        try (InferenceSession source = sharing.openSession()) {
          held.prefill(source, documentTokens, 0);
          prefix = sharing.freezePrefix(source);
        }
        double prefillSeconds = (System.nanoTime() - a) / 1e9;

        long b = System.nanoTime();
        try (InferenceSession branch = sharing.fork(prefix)) {
          int position = documentTokens.length;
          int[] head = new int[tail.length - 1];
          System.arraycopy(tail, 0, head, 0, head.length);
          held.prefill(branch, head, position);
          sharing.forwardHiddenState(branch, tail[tail.length - 1], position + tail.length - 1);
        }
        double tailSeconds = (System.nanoTime() - b) / 1e9;

        // Ten tails the old way against ten tails in one ragged pass, measured back to back in the
        // same round so neither arm can be handed a warmer machine than the other.
        long c = System.nanoTime();
        for (int q = 0; q < TAIL_COUNT; q++) {
          try (InferenceSession branch = sharing.fork(prefix)) {
            int position = documentTokens.length;
            int[] head = new int[tail.length - 1];
            System.arraycopy(tail, 0, head, 0, head.length);
            held.prefill(branch, head, position);
            sharing.forwardHiddenState(branch, tail[tail.length - 1], position + tail.length - 1);
          }
        }
        double sequentialTails = (System.nanoTime() - c) / 1e9;

        int[][] tails = new int[TAIL_COUNT][];
        int[] starts = new int[TAIL_COUNT];
        for (int q = 0; q < TAIL_COUNT; q++) {
          tails[q] = tail;
          starts[q] = documentTokens.length;
        }
        InferenceSession[] branches = new InferenceSession[TAIL_COUNT];
        long d = System.nanoTime();
        try {
          for (int q = 0; q < TAIL_COUNT; q++) {
            branches[q] = sharing.fork(prefix);
          }
          held.prefillBatchHiddenStates(branches, tails, starts);
        } finally {
          for (InferenceSession branch : branches) {
            if (branch != null) {
              branch.close();
            }
          }
        }
        double batchedTails = (System.nanoTime() - d) / 1e9;

        if (round >= WARMUP) {
          prefillBest = Math.min(prefillBest, prefillSeconds);
          tailBest = Math.min(tailBest, tailSeconds);
          sequentialBest = Math.min(sequentialBest, sequentialTails);
          batchedBest = Math.min(batchedBest, batchedTails);
        }
        System.out.printf(
            "  round %d%s  prefill %7.3f s  tail %6.3f s   %d tails: seq %6.3f s  batched %6.3f s%n",
            round + 1,
            round < WARMUP ? " (warmup)" : "        ",
            prefillSeconds,
            tailSeconds,
            TAIL_COUNT,
            sequentialTails,
            batchedTails);
      }

      System.out.printf("%n  best prefill        %7.3f s  (%6.2f ms/token)%n",
          prefillBest, prefillBest * 1000 / documentTokens.length);
      System.out.printf("  best question tail  %7.3f s  (%6.2f ms/token)%n",
          tailBest, tailBest * 1000 / tail.length);
      System.out.printf("  %d tails sequential  %7.3f s%n", TAIL_COUNT, sequentialBest);
      System.out.printf("  %d tails batched     %7.3f s   speedup %.2fx%n",
          TAIL_COUNT, batchedBest, sequentialBest / batchedBest);
      System.out.printf("  ten questions, seq  %7.3f s%n", prefillBest + sequentialBest);
      System.out.printf("  ten questions, bat  %7.3f s%n%n", prefillBest + batchedBest);
    }
  }

  private static String widthOf(PureJavaBackend backend, SharedPrefixInferenceBackend sharing) {
    try (InferenceSession probe = sharing.openSession()) {
      return String.valueOf(sharing.forwardHiddenState(probe, 1, 0).length);
    } catch (RuntimeException unsupported) {
      return "unknown";
    }
  }
}
