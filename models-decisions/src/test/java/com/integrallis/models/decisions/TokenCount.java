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

import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Exact input-token counts per decision, for the JevBench Cost axis.
 *
 * <p>Cost for a self-hosted entrant is a size-class list price times <em>input tokens per
 * decision</em>. Estimating those tokens from characters put the axis within a few points of the
 * truth by luck, not by method; this counts them with the tokenizer that actually ran.
 *
 * <p>Both arms are reported because they feed the model different text: the letter arm appends the
 * rendered option block and does one forward pass, the candidate arm scores each option as a
 * continuation of one shared prefix, so its input is the prompt plus the option tokens.
 */
public final class TokenCount {

  private TokenCount() {}

  public static void main(String[] args) throws Exception {
    Path model = Path.of(args[0]);
    Path tasks = Path.of(args[1]);
    // Under rawPrompt the fourth column is already the complete prompt, as the runner feeds it,
    // so the count is the prompt itself and no option block is appended.
    boolean rawPrompt = args.length > 2 && Boolean.parseBoolean(args[2]);
    List<String[]> rows = new ArrayList<>();
    for (String line : Files.readAllLines(tasks, StandardCharsets.UTF_8)) {
      if (!line.isBlank()) {
        rows.add(line.split("\t", -1));
      }
    }
    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      long letterTotal = 0;
      long candidateTotal = 0;
      System.out.println("id\tletterTokens\tcandidateTokens");
      for (String[] row : rows) {
        String id = row[0];
        List<String> labels = List.of(row[2].split("\\|", -1));
        String prompt = row[3].replace("\\n", "\n");

        int letters =
            rawPrompt
                ? backend.tokenizer().encode(prompt).length
                : backend
                    .tokenizer()
                    .encode(prompt + "\n" + LetterLogitScorer.renderOptions(labels))
                    .length;
        int candidate = backend.tokenizer().encode(prompt).length;
        for (String label : labels) {
          candidate += backend.tokenizer().encode(" " + label).length;
        }
        letterTotal += letters;
        candidateTotal += candidate;
        System.out.printf("%s\t%d\t%d%n", id, letters, candidate);
      }
      System.err.printf(
          "decisions=%d  letter-arm mean %.1f tokens/decision  candidate-arm mean %.1f%n",
          rows.size(), (double) letterTotal / rows.size(), (double) candidateTotal / rows.size());
    }
  }
}
