package demo;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.Tokenizer;
import org.modeljars.ModelJarDecisionRuntime;
import org.modeljars.composite.harriet.Harriet;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Is a question's prefill compute bound or bandwidth bound?
 *
 * <p>It decides whether answering questions together can ever be cheaper than answering them one
 * at a time. If prefill is bandwidth bound then its cost is one read of the weights and riding
 * more tokens along is free, which is what batching questions exploits. If it is compute bound
 * then N questions cost N questions' arithmetic however they are arranged, and batching saves
 * nothing. Time per token against thread count separates them: compute scales with threads, a
 * memory wall does not.
 */
public final class Threads {
  public static void main(String... args) throws Exception {
    String evidence = Files.readString(Path.of(args[0])).strip();
    try (ModelJarDecisionRuntime runtime = Harriet.open()) {
      InferenceBackend backend = runtime.backend();
      Tokenizer tokenizer = backend.tokenizer();
      int[] tokens = tokenizer.encode(evidence);
      System.out.println();
      System.out.printf("  threads %s   tokens %d%n", System.getProperty("models.native.threads", "default"), tokens.length);

      double best = Double.MAX_VALUE;
      for (int i = 0; i < 4; i++) {
        backend.reset();
        long t = System.nanoTime();
        backend.prefill(tokens, 0);
        best = Math.min(best, (System.nanoTime() - t) / 1e9);
      }
      System.out.printf("  batched prefill    %7.3f s   %6.2f ms/token%n", best, 1000 * best / tokens.length);

      double decode = Double.MAX_VALUE;
      for (int i = 0; i < 4; i++) {
        backend.reset();
        backend.prefill(Arrays.copyOf(tokens, tokens.length - 1), 0);
        long t = System.nanoTime();
        backend.forward(tokens[tokens.length - 1], tokens.length - 1);
        decode = Math.min(decode, (System.nanoTime() - t) / 1e9);
      }
      System.out.printf("  single-token step  %7.3f s%n%n", decode);
    }
  }
}
