package demo;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ResumableInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.decisions.LetterLogitScorer;
import com.integrallis.models.decisions.Noul;
import org.modeljars.ModelJarDecisionRuntime;
import org.modeljars.composite.harriet.Harriet;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * A warm decision costs 0.486 s and only 0.333 s of that is arithmetic. Where is the rest?
 *
 * <p>The suffix is 18 tokens and a batched prefill measures 18.5 ms a token, compute bound and at
 * this machine's ceiling. That accounts for two thirds. The remaining third is not arithmetic, and a
 * third of a decision spent on bookkeeping is worth more attention than another pass at the kernel.
 *
 * <p>Each stage is timed on its own, in the order the runtime performs them, so the parts add up to
 * the whole rather than being inferred from a difference.
 */
public final class Where {
  public static void main(String... args) throws Exception {
    String evidence = Files.readString(Path.of(args[0])).strip();
    List<String> criteria =
        List.of(
            "Is a monthly fee stated?",
            "Is an uptime guarantee stated?",
            "May Customer Data be used to train models?",
            "Is a liability cap stated?",
            "Can the agreement be terminated for convenience?",
            "Is the governing law stated?",
            "Is arbitration required?",
            "Is a renewal term stated?");

    try (ModelJarDecisionRuntime runtime = Harriet.open()) {
      InferenceBackend backend = runtime.backend();
      Tokenizer tokenizer = backend.tokenizer();
      ResumableInferenceBackend resumable = (ResumableInferenceBackend) backend;

      int[] shared = tokenizer.encode(evidence);
      int[][] suffixes = new int[criteria.size()][];
      for (int index = 0; index < criteria.size(); index++) {
        int[] whole =
            tokenizer.encode(
                evidence
                    + "\n"
                    + criteria.get(index)
                    + "\n"
                    + LetterLogitScorer.renderOptions(List.of("false", "true")));
        suffixes[index] = Arrays.copyOfRange(whole, shared.length, whole.length);
      }

      for (int round = 0; round < 3; round++) {
        for (String criterion : criteria) {
          runtime.decide(new Noul(criterion), evidence);
        }
      }

      backend.reset();
      backend.prefill(shared, 0);
      ResumableInferenceBackend.Resumption point = resumable.capture();

      long tokenize = 0;
      long resume = 0;
      long prefill = 0;
      long score = 0;
      int rounds = 4;
      for (int round = 0; round < rounds; round++) {
        for (int index = 0; index < criteria.size(); index++) {
          long t = System.nanoTime();
          tokenizer.encode(
              evidence + "\n" + criteria.get(index) + "\n"
                  + LetterLogitScorer.renderOptions(List.of("false", "true")));
          tokenize += System.nanoTime() - t;

          t = System.nanoTime();
          resumable.resume(point);
          resume += System.nanoTime() - t;

          t = System.nanoTime();
          float[] logits = backend.prefill(suffixes[index], shared.length);
          prefill += System.nanoTime() - t;

          t = System.nanoTime();
          new LetterLogitScorer(1.0)
              .score(
                  new Noul(criteria.get(index)),
                  logits,
                  new int[] {
                    tokenizer.encode(" A")[tokenizer.encode(" A").length - 1],
                    tokenizer.encode(" B")[tokenizer.encode(" B").length - 1]
                  });
          score += System.nanoTime() - t;
        }
      }

      int n = rounds * criteria.size();
      double total = (tokenize + resume + prefill + score) / 1e9 / n;
      System.out.printf("%n  suffix tokens %d, %d decisions timed%n%n", suffixes[0].length, n);
      line("tokenize the prompt", tokenize, n, total);
      line("resume the shared prefix", resume, n, total);
      line("prefill the suffix and read logits", prefill, n, total);
      line("score the letters", score, n, total);
      System.out.printf("  %-36s %8.4f s%n%n", "sum", total);
    }
  }

  private static void line(String label, long nanos, int n, double total) {
    double seconds = nanos / 1e9 / n;
    System.out.printf("  %-36s %8.4f s   %5.1f%%%n", label, seconds, 100 * seconds / total);
  }
}
