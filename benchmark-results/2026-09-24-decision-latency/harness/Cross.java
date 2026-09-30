package demo;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * IS THE DIVERGENCE ABOUT BATCHING, OR IS THE MODEL JUST THIS SENSITIVE?
 *
 * <p>A single matmul measures 3e-7 relative between the two kernels, deterministic and consistent
 * across batch sizes -- ordinary fp32 rounding. Whole-model logits diverge by 0.07 on average, four
 * orders of magnitude more. Either something in the graph amplifies rounding that hard, or the
 * matmul test missed the real defect.
 *
 * <p>Three comparisons separate those. Native against Java at the <em>same</em> batch: if that is
 * also 0.07, the divergence has nothing to do with batching and everything to do with two kernels
 * rounding differently in a graph that amplifies. Native batched against native stepped: the
 * measured symptom. And a deliberate one-ulp nudge to a single embedding element, run through the
 * Java path alone: an upper bound on how much a rounding-sized perturbation can grow by itself.
 */
public final class Cross {
  private static final int TOKENS = 24;

  public static void main(String... args) throws Exception {
    Path model = Path.of(args[0]);
    int[] tokens = new int[TOKENS];
    java.util.Random random = new java.util.Random(20260924L);
    for (int index = 0; index < tokens.length; index++) {
      tokens[index] = 1000 + random.nextInt(20000);
    }

    try (PureJavaBackend java = PureJavaBackend.load(model, GgufBatchedMatrixKernel.none());
        PureJavaBackend rust = PureJavaBackend.load(model, RustGgufBatchedMatrixKernel.openBundled())) {

      // Warm each shape on each backend. An unwarmed comparison measures the JIT, not the kernel.
      int warmup = Integer.getInteger("diag.warmup", 8);
      for (int index = 0; index < warmup; index++) {
        batched(java, tokens);
        stepped(java, tokens);
        split(java, tokens);
        batched(rust, tokens);
        stepped(rust, tokens);
        split(rust, tokens);
      }
      System.out.printf("  warmup %d rounds of every shape on both backends%n", warmup);

      float[] javaBatched = batched(java, tokens);
      float[] javaStepped = stepped(java, tokens);
      float[] rustBatched = batched(rust, tokens);
      float[] rustStepped = stepped(rust, tokens);

      System.out.printf("%n  %d tokens, logits at the final position%n%n", TOKENS);
      report("java batched vs java stepped   (control)", javaBatched, javaStepped);
      report("rust batched vs rust stepped   (symptom)", rustBatched, rustStepped);
      report("rust stepped vs java stepped   (same shape, two kernels)", rustStepped, javaStepped);
      report("rust batched vs java batched   (same shape, two kernels)", rustBatched, javaBatched);

      // Same kernel, same tokens, same arithmetic in exact terms -- only the grouping of the
      // prefill changes. Nothing here is a different implementation or a different precision, so
      // whatever this produces is the model's own sensitivity to a scheduling decision.
      report("rust one batch vs rust two batches (schedule only)", rustBatched, split(rust, tokens));
      report("java one batch vs java two batches (schedule only)", javaBatched, split(java, tokens));
      System.out.println();
    }
  }

  private static float[] batched(PureJavaBackend backend, int[] tokens) {
    backend.reset();
    try (InferenceSession session = backend.openSession()) {
      backend.prefill(session, Arrays.copyOf(tokens, tokens.length - 1), 0);
      return backend
          .forward(session, tokens[tokens.length - 1], tokens.length - 1)
          .clone();
    }
  }

  /** The same prefill, cut in half. Mathematically the identical computation. */
  private static float[] split(PureJavaBackend backend, int[] tokens) {
    backend.reset();
    int half = (tokens.length - 1) / 2;
    try (InferenceSession session = backend.openSession()) {
      backend.prefill(session, Arrays.copyOf(tokens, half), 0);
      backend.prefill(session, Arrays.copyOfRange(tokens, half, tokens.length - 1), half);
      return backend.forward(session, tokens[tokens.length - 1], tokens.length - 1).clone();
    }
  }

  private static float[] stepped(PureJavaBackend backend, int[] tokens) {
    backend.reset();
    try (InferenceSession session = backend.openSession()) {
      float[] logits = null;
      for (int index = 0; index < tokens.length; index++) {
        logits = backend.forward(session, tokens[index], index);
      }
      return logits.clone();
    }
  }

  private static void report(String label, float[] left, float[] right) {
    double worst = 0;
    double total = 0;
    double scale = 0;
    for (int index = 0; index < left.length; index++) {
      worst = Math.max(worst, Math.abs(left[index] - right[index]));
      total += Math.abs(left[index] - right[index]);
      scale = Math.max(scale, Math.abs(right[index]));
    }
    System.out.printf(
        "  %-58s max %.3e  mean %.3e  relative %.3e  argmax %s%n",
        label,
        worst,
        total / left.length,
        scale == 0 ? worst : worst / scale,
        argmax(left) == argmax(right) ? "same" : "DIFFERS");
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }
}
