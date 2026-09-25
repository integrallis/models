package demo;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Path;
import java.util.Random;

/**
 * Prefill cost against the width it is batched at, and whether the answer survives the change.
 *
 * <p>A decision is 98.5% prefill (MEASURED 2026-09-24: tokenise 1.2 ms, resume 4.6 ms, score 0.2 ms,
 * prefill 385 ms), and the Q4_K kernel is within about 10% of its cold ceiling, so per-token
 * arithmetic is spent. What is left is how many tokens ride in one pass. The default is 32
 * ({@code PureJavaPlanConfiguration.DEFAULT_PREFILL_BATCH_SIZE}), so a 60-token prompt costs two
 * passes and the second one carries 28 tokens of useful work in a batch sized for 32.
 *
 * <p>Wider batches raise arithmetic intensity: Q4_K spends 144 bytes per 256 weights, so every
 * weight byte feeds {@code 32 * batch / 18} multiply-accumulates. If the rate rises with width then
 * covering a prompt in one pass is free latency.
 *
 * <p><b>It is only free if the answer does not move.</b> This project has already shipped a defect
 * where a grouped read differed from a single read by up to 0.28 of a logit, because a lone question
 * reads its answer out of a batch of one row and a group reads its out of a batch of many. So this
 * harness prints a checksum of the final-position logits alongside the timing, and the sweep script
 * compares them across widths. A latency win that changes answers is not a latency win.
 *
 * <p>One JVM per width: the batch size is read from a system property at load, and mixing widths in
 * one process would also mix JIT states.
 */
public final class PrefillWidth {
  public static void main(String... args) throws Exception {
    Path model = Path.of(args[0]);
    int tokens = Integer.getInteger("diag.tokens", 60);
    int rounds = Integer.getInteger("diag.rounds", 6);
    int warmups = Integer.getInteger("diag.warmups", 4);

    try (RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.openBundled();
        PureJavaBackend backend = PureJavaBackend.load(model, kernel)) {
      int[] prompt = new int[tokens];
      Random random = new Random(11L);
      for (int index = 0; index < tokens; index++) {
        prompt[index] = 1000 + random.nextInt(20000);
      }

      // Warm the path being timed, not some other one. An unwarmed Panama vector kernel takes a
      // different route with a different accumulation order; NOTES.md section 0 is what that cost.
      for (int warm = 0; warm < warmups; warm++) {
        backend.reset();
        try (InferenceSession session = backend.openSession()) {
          backend.prefill(session, prompt, 0);
        }
      }

      double best = Double.MAX_VALUE;
      double checksum = 0;
      double maximumAbsolute = 0;
      for (int round = 0; round < rounds; round++) {
        backend.reset();
        try (InferenceSession session = backend.openSession()) {
          long started = System.nanoTime();
          float[] logits = backend.prefill(session, prompt, 0);
          double seconds = (System.nanoTime() - started) / 1e9;
          best = Math.min(best, seconds);
          // Order-independent so the checksum tests the values and not the traversal, and scaled
          // per index so two different vectors cannot cancel to the same total.
          double sum = 0;
          double peak = 0;
          for (int index = 0; index < logits.length; index++) {
            sum += (double) logits[index] * (index % 1021 + 1);
            peak = Math.max(peak, Math.abs(logits[index]));
          }
          checksum = sum;
          maximumAbsolute = peak;
        }
      }

      System.out.printf(
          "  width %-4s tokens %-4d %.4f s  %.2f ms/token  checksum %.6f  max|logit| %.6f%n",
          System.getProperty("models.purejava.prefillBatchSize", "default(32)"),
          tokens,
          best,
          1000 * best / tokens,
          checksum,
          maximumAbsolute);
    }
  }
}
