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
import java.util.ArrayList;
import java.util.List;

/**
 * Many decisions against one document, timed so the shape of the cost is visible.
 *
 * <p>A hosted decision API is billed and timed per call, and each call carries the document again.
 * Here the document is prefilled once and frozen, and every question forks that physical prefix, so
 * the second question and the twentieth cost only their own tokens. This prints the split — the
 * one-off prefill against the marginal cost per question — because the average alone hides which
 * of the two is being paid.
 *
 * <p>The prefix sharing is asserted rather than assumed. A fork that quietly copied the prefix
 * would produce identical answers and a completely different cost curve, which is exactly the kind
 * of difference an average would swallow.
 */
public final class BriefingDemo {

  private BriefingDemo() {}

  /** Arguments: artifact, GGUF base, document file, questions file (one per line). */
  public static void main(String[] args) throws IOException {
    DecisionArtifact artifact = DecisionArtifact.read(Path.of(args[0]));
    Path model = Path.of(args[1]);
    String document = Files.readString(Path.of(args[2]), StandardCharsets.UTF_8).strip();
    List<String> questions =
        Files.readAllLines(Path.of(args[3]), StandardCharsets.UTF_8).stream()
            .map(String::strip)
            .filter(line -> !line.isEmpty())
            .toList();

    // Checked before any inference: a wrong base wastes the whole run and looks like a bad model.
    artifact.requireBase(model);

    PureJavaBackend backend;
    String kernel;
    try {
      backend = PureJavaBackend.load(model, RustGgufBatchedMatrixKernel.openBundled());
      kernel = "rust-ffm";
    } catch (RuntimeException | UnsatisfiedLinkError unavailable) {
      backend = PureJavaBackend.load(model);
      kernel = "pure-java fallback";
    }

    try (PureJavaBackend held = backend) {
      SharedPrefixInferenceBackend sharing = held;
      int[] documentTokens = held.tokenizer().encode(document + "\n\nQuestion:");

      System.out.println();
      System.out.println("  Integrallis Decisions — one document, many questions");
      System.out.printf("  base %s, kernel %s%n", artifact.baseModel(), kernel);
      System.out.printf(
          "  document %d tokens, %d questions%n%n", documentTokens.length, questions.size());

      // One prefill, paid once, for every question that follows.
      long prefillStart = System.nanoTime();
      SharedInferencePrefix prefix;
      try (InferenceSession source = sharing.openSession()) {
        held.prefill(source, documentTokens, 0);
        prefix = sharing.freezePrefix(source);
      }
      double prefillSeconds = (System.nanoTime() - prefillStart) / 1e9;
      System.out.printf("  prefill (once)              %7.3f s%n%n", prefillSeconds);

      List<Double> marginal = new ArrayList<>();
      List<Double> probabilities = new ArrayList<>();
      boolean sharedEverywhere = true;
      for (int index = 0; index < questions.size(); index++) {
        String question = questions.get(index);
        int[] tail =
            held.tokenizer()
                .encode(
                    " "
                        + question
                        + "\nIs the question answerable from the text above? Answer yes or no."
                        + "\nAnswer:");

        long start = System.nanoTime();
        double probability;
        try (InferenceSession branch = sharing.fork(prefix)) {
          int position = documentTokens.length;
          if (tail.length > 1) {
            int[] head = new int[tail.length - 1];
            System.arraycopy(tail, 0, head, 0, head.length);
            held.prefill(branch, head, position);
          }
          float[] hidden =
              sharing.forwardHiddenState(branch, tail[tail.length - 1], position + tail.length - 1);
          probability = artifact.decide(hidden).probabilityOfTrue();

          // A second fork witnesses that the branch really sits on the shared storage.
          try (InferenceSession witness = sharing.fork(prefix)) {
            sharedEverywhere &= sharing.sharesPrefixStorage(branch, witness);
          }
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        marginal.add(seconds);
        probabilities.add(probability);

        System.out.printf(
            "  Q%-2d %-46s %s  p=%.3f  %6.3f s%n",
            index + 1,
            question.length() > 46 ? question.substring(0, 43) + "..." : question,
            probability >= 0.5 ? "YES" : "NO ",
            probability,
            seconds);
      }

      double total = prefillSeconds + marginal.stream().mapToDouble(Double::doubleValue).sum();
      double meanMarginal =
          marginal.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
      System.out.println();
      System.out.printf("  prefix shared by every fork  %s%n", sharedEverywhere ? "yes" : "NO");
      System.out.printf("  marginal cost per question   %7.3f s%n", meanMarginal);
      System.out.printf("  total for %-2d questions       %7.3f s%n", questions.size(), total);
      System.out.printf("  effective per question       %7.3f s%n", total / questions.size());
      System.out.println();

      // The machine-readable line, so the comparison is not retyped from a screen.
      Path out = Path.of(args.length > 4 ? args[4] : "decisions-timing.json");
      StringBuilder json = new StringBuilder();
      json.append("{\"system\":\"integrallis-decisions\",\"kernel\":\"").append(kernel);
      json.append("\",\"document_tokens\":").append(documentTokens.length);
      json.append(",\"questions\":").append(questions.size());
      json.append(",\"prefill_s\":").append(String.format("%.4f", prefillSeconds));
      json.append(",\"marginal_s\":[");
      for (int i = 0; i < marginal.size(); i++) {
        json.append(i == 0 ? "" : ",").append(String.format("%.4f", marginal.get(i)));
      }
      json.append("],\"probabilities\":[");
      for (int i = 0; i < probabilities.size(); i++) {
        json.append(i == 0 ? "" : ",").append(String.format("%.4f", probabilities.get(i)));
      }
      json.append("],\"total_s\":").append(String.format("%.4f", total));
      json.append(",\"prefix_shared\":").append(sharedEverywhere).append("}");
      Files.writeString(out, json.toString(), StandardCharsets.UTF_8);
      System.out.printf("  timings written to %s%n%n", out);
    }
  }
}
