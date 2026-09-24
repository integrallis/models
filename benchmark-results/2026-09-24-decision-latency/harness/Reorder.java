package demo;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.decisions.LetterLogitScorer;
import org.modeljars.ModelJarDecisionRuntime;
import org.modeljars.composite.harriet.Harriet;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * What if the lettered options came before the criterion instead of after it?
 *
 * <p>A decision's cost is the arithmetic of the tokens that follow the shared evidence: MEASURED at
 * 18.5 ms each, linear, compute bound. Today those tokens are the criterion AND the rendered
 * options, and for a true/false question the options are most of them -- the criterion is about six
 * tokens and the options block about eleven. Every question re-reads an options block identical to
 * every other question's.
 *
 * <p>Putting the options before the criterion makes them part of the shared prefix, so a question
 * costs only its own words. That is a change to the prompt and so to the answers, which is why this
 * measures agreement as well as time: a prompt that is twice as fast and disagrees with the
 * qualified one is not a speedup, it is a different model.
 */
public final class Reorder {
  public static void main(String... args) throws Exception {
    List<String[]> cases = new ArrayList<>();
    for (String line : Files.readAllLines(Path.of(args[0]))) {
      String[] fields = line.split("\t", -1);
      if (fields.length == 5 && !fields[0].equals("id")) {
        cases.add(fields);
      }
    }

    try (ModelJarDecisionRuntime runtime = Harriet.open()) {
      InferenceBackend backend = runtime.backend();
      Tokenizer tokenizer = backend.tokenizer();

      // Warm the machine before anything is timed.
      for (int index = 0; index < 3; index++) {
        answer(backend, tokenizer, cases.get(0), false);
      }

      System.out.println();
      System.out.printf(
          "  %-18s %-7s %-9s %-9s %-9s %-9s %-7s%n",
          "case", "options", "now tok", "new tok", "now s", "new s", "agree");

      int agreed = 0;
      double nowTotal = 0;
      double newTotal = 0;
      int nowTokens = 0;
      int newTokens = 0;
      for (String[] one : cases) {
        Answer now = answer(backend, tokenizer, one, false);
        Answer reordered = answer(backend, tokenizer, one, true);
        boolean agree = now.winner == reordered.winner;
        agreed += agree ? 1 : 0;
        nowTotal += now.seconds;
        newTotal += reordered.seconds;
        nowTokens += now.suffixTokens;
        newTokens += reordered.suffixTokens;
        System.out.printf(
            "  %-18s %-7d %-9d %-9d %-9.3f %-9.3f %-7s%n",
            one[0],
            one[2].split("\\|", -1).length,
            now.suffixTokens,
            reordered.suffixTokens,
            now.seconds,
            reordered.seconds,
            agree ? "yes" : "NO");
      }
      System.out.println();
      System.out.printf(
          "  %d cases   suffix tokens %d -> %d   mean %.3f s -> %.3f s   %.2fx   winners agree %d/%d%n%n",
          cases.size(),
          nowTokens,
          newTokens,
          nowTotal / cases.size(),
          newTotal / cases.size(),
          nowTotal / newTotal,
          agreed,
          cases.size());
    }
  }

  private record Answer(int winner, double seconds, int suffixTokens) {}

  /**
   * Answers one case, timing only what a warm runtime would pay: the question's own tokens.
   *
   * <p>The shared prefix is prefilled outside the clock in both arrangements, because in both it is
   * read once for a batch of questions and resumed thereafter. What is being compared is the part
   * that is genuinely per question.
   */
  private static Answer answer(
      InferenceBackend backend, Tokenizer tokenizer, String[] one, boolean optionsFirst) {
    List<String> labels = List.of(one[2].split("\\|", -1));
    String options = LetterLogitScorer.renderOptions(labels);
    String evidence = one[4];
    String criterion = one[3];

    String prefix = optionsFirst ? evidence + "\n" + optionsBlock(options) : evidence;
    String whole =
        optionsFirst
            ? prefix + "\n" + criterion + "\nAnswer:"
            : evidence + "\n" + criterion + "\n" + options;

    int[] prefixTokens = tokenizer.encode(prefix);
    int[] prompt = tokenizer.encode(whole);
    int last = prompt.length - 1;

    backend.reset();
    backend.prefill(prefixTokens, 0);

    long started = System.nanoTime();
    if (last > prefixTokens.length) {
      backend.prefill(Arrays.copyOfRange(prompt, prefixTokens.length, last), prefixTokens.length);
    }
    float[] logits = backend.forward(prompt[last], last);
    double seconds = (System.nanoTime() - started) / 1e9;

    int winner = 0;
    float best = Float.NEGATIVE_INFINITY;
    for (int index = 0; index < labels.size(); index++) {
      int[] encoded = tokenizer.encode(" " + (char) ('A' + index));
      float value = logits[encoded[encoded.length - 1]];
      if (value > best) {
        best = value;
        winner = index;
      }
    }
    return new Answer(winner, seconds, prompt.length - prefixTokens.length);
  }

  /** The rendered options without the trailing answer cue, which stays at the end of the prompt. */
  private static String optionsBlock(String options) {
    return options.substring(0, options.length() - "Answer:".length()).stripTrailing();
  }
}
