package demo;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.decisions.AnswerSpace;
import com.integrallis.models.decisions.LetterLogitScorer;
import com.integrallis.models.decisions.Noul;
import org.modeljars.ModelJarDecisionRuntime;
import org.modeljars.composite.harriet.Harriet;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * How many weight sweeps does a decision cost, and how many does a grouped decision cost?
 *
 * <p>A question's tokens sit after the evidence, so a decision is one prefill of the shared
 * evidence, one prefill of the question, and one final forward. A grouped decision instead walks
 * the questions forward in lockstep, one token from each branch per step. If every step sweeps the
 * weights, a group costs as many sweeps as the longest question has tokens, and the suffix token
 * count below should predict the grouped time.
 */
public final class Split {
  public static void main(String... args) throws Exception {
    String evidence = Files.readString(Path.of(args[0])).strip();
    List<String> labels = List.of("true", "false");
    String options = LetterLogitScorer.renderOptions(labels);

    String short_ = "Is a fee stated?";
    String medium = "Considering the payment terms and the billing schedule set out above, is a monthly fee stated?";
    StringBuilder builder = new StringBuilder("Considering the payment terms");
    for (int i = 0; i < 6; i++) {
      builder.append(", the billing schedule, the invoicing cadence and any stated discounts");
    }
    builder.append(", is a monthly fee stated?");
    String long_ = builder.toString();

    try (ModelJarDecisionRuntime runtime = Harriet.open()) {
      InferenceBackend backend = runtime.backend();
      Tokenizer tokenizer = backend.tokenizer();
      // Warm every shape that gets timed below, at each criterion length.
      for (int round = 0; round < 4; round++) {
        for (String criterion : new String[] {short_, medium, long_}) {
          Harriet.noul(runtime, criterion, evidence);
        }
      }

      int[] evidenceTokens = tokenizer.encode(evidence);
      System.out.println();
      System.out.printf("  evidence tokens %d%n", evidenceTokens.length);
      System.out.println();
      System.out.printf(
          "  %-8s %-8s %-11s %-11s %-11s %-11s %-11s%n",
          "criterion", "suffix", "prefix s", "suffix s", "forward s", "alone s", "two grouped s");

      for (String[] entry : new String[][] {{"short", short_}, {"medium", medium}, {"long", long_}}) {
        int[] prompt = tokenizer.encode(evidence + "\n" + entry[1] + "\n" + options);
        int suffix = prompt.length - evidenceTokens.length;

        backend.reset();
        long t = System.nanoTime();
        backend.prefill(Arrays.copyOf(prompt, evidenceTokens.length), 0);
        double prefix = (System.nanoTime() - t) / 1e9;

        t = System.nanoTime();
        backend.prefill(
            Arrays.copyOfRange(prompt, evidenceTokens.length, prompt.length - 1),
            evidenceTokens.length);
        double suffixSeconds = (System.nanoTime() - t) / 1e9;

        t = System.nanoTime();
        backend.forward(prompt[prompt.length - 1], prompt.length - 1);
        double forward = (System.nanoTime() - t) / 1e9;

        t = System.nanoTime();
        Harriet.noul(runtime, entry[1], evidence);
        double alone = (System.nanoTime() - t) / 1e9;

        List<AnswerSpace> spaces = new ArrayList<>();
        spaces.add(new Noul(entry[1]));
        spaces.add(new Noul(entry[1]));
        t = System.nanoTime();
        Harriet.decideAll(runtime, spaces, evidence);
        double grouped = (System.nanoTime() - t) / 1e9;

        System.out.printf(
            "  %-8s %-8d %-11.3f %-11.3f %-11.3f %-11.3f %-11.3f%n",
            entry[0], suffix, prefix, suffixSeconds, forward, alone, grouped);
      }
      System.out.println();
    }
  }
}
