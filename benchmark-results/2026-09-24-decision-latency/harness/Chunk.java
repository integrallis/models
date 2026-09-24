package demo;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.decisions.LetterLogitScorer;
import org.modeljars.ModelJarDecisionRuntime;
import org.modeljars.composite.harriet.Harriet;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Does prefilling a question as one batch give the same answer as feeding it a token at a time?
 *
 * <p>A batched prefill runs the recurrence as a chunked associative scan: the same recurrence
 * algebraically, different arithmetic. A grouped decision walks its branches one token at a time
 * and so takes the exact recurrence instead. If the two disagree then a grouped answer and a
 * one-at-a-time answer disagree for that reason alone, and the size of the disagreement says how
 * much the chunked scan moves an answer.
 */
public final class Chunk {
  public static void main(String... args) throws Exception {
    String evidence = Files.readString(Path.of(args[0])).strip();
    String[] criteria = {
      "Is a monthly fee stated?",
      "Is an uptime guarantee stated?",
      "May Customer Data be used to train models?",
      "Is a liability cap stated?",
      "Does the liability cap apply to data breaches?",
      "Can the agreement be terminated for convenience?"
    };
    List<String> labels = List.of("true", "false");
    String options = LetterLogitScorer.renderOptions(labels);

    try (ModelJarDecisionRuntime runtime = Harriet.open()) {
      InferenceBackend backend = runtime.backend();
      Tokenizer tokenizer = backend.tokenizer();
      int[] letters = new int[labels.size()];
      for (int index = 0; index < labels.size(); index++) {
        int[] encoded = tokenizer.encode(" " + (char) ('A' + index));
        letters[index] = encoded[encoded.length - 1];
      }
      int[] evidenceTokens = tokenizer.encode(evidence);

      // Warm both shapes. An unwarmed comparison of two code paths measures which one the JIT got
      // to first, and both of these round differently before they are compiled.
      int[] warm = tokenizer.encode(evidence + "\n" + criteria[0] + "\n" + options);
      for (int round = 0; round < 6; round++) {
        backend.reset();
        backend.prefill(Arrays.copyOf(warm, warm.length - 1), 0);
        backend.forward(warm[warm.length - 1], warm.length - 1);
        backend.reset();
        backend.prefill(Arrays.copyOf(warm, evidenceTokens.length), 0);
        for (int index = evidenceTokens.length; index < warm.length; index++) {
          backend.forward(warm[index], index);
        }
      }

      System.out.println();
      System.out.printf(
          "  %-46s %-7s %-10s %-10s %-10s %-11s%n",
          "criterion", "suffix", "batched", "per token", "shift", "max logit");

      double worst = 0;
      for (String criterion : criteria) {
        int[] prompt = tokenizer.encode(evidence + "\n" + criterion + "\n" + options);
        int last = prompt.length - 1;
        int[] suffix = Arrays.copyOfRange(prompt, evidenceTokens.length, last);

        backend.reset();
        backend.prefill(Arrays.copyOf(prompt, evidenceTokens.length), 0);
        if (suffix.length > 0) {
          backend.prefill(suffix, evidenceTokens.length);
        }
        float[] batched = backend.forward(prompt[last], last).clone();

        backend.reset();
        backend.prefill(Arrays.copyOf(prompt, evidenceTokens.length), 0);
        for (int index = 0; index < suffix.length; index++) {
          backend.forward(suffix[index], evidenceTokens.length + index);
        }
        float[] stepped = backend.forward(prompt[last], last).clone();

        double gap = 0;
        for (int index = 0; index < batched.length; index++) {
          gap = Math.max(gap, Math.abs(batched[index] - stepped[index]));
        }
        double first = probability(batched, letters);
        double second = probability(stepped, letters);
        worst = Math.max(worst, Math.abs(first - second));
        System.out.printf(
            "  %-46s %-7d %-10.4f %-10.4f %-10.4f %-11.3e%n",
            criterion, suffix.length + 1, first, second, Math.abs(first - second), gap);
      }
      System.out.printf("%n  worst probability shift  %.4f%n%n", worst);
    }
  }

  private static double probability(float[] logits, int[] letters) {
    double highest = Double.NEGATIVE_INFINITY;
    for (int letter : letters) {
      highest = Math.max(highest, logits[letter]);
    }
    double total = 0;
    for (int letter : letters) {
      total += Math.exp(logits[letter] - highest);
    }
    return Math.exp(logits[letters[0]] - highest) / total;
  }
}
