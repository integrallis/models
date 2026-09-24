package demo;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * PREFILLING N TOKENS MUST EQUAL STEPPING N TOKENS.
 *
 * <p>A batch is a scheduling decision. It changes which kernel runs and in what order things
 * accumulate; it must not change the answer. MEASURED 2026-09-24 on the shipped Harriet base: the
 * same question prefilled as one batch and fed a token at a time disagreed by up to 0.044 of
 * probability, with max logit gaps of 0.5 to 0.8. That is far too large to be fp32 reordering over
 * a 2560-wide dot product, so something else is going on and this finds out what.
 *
 * <p>Bisects by batch size. Divergence that appears at two tokens is the batched matrix kernel;
 * divergence that appears only at eight is the chunked Gated DeltaNet scan, whose minimum chunk is
 * eight. Run with and without the native kernel to separate Rust from Java.
 */
public final class Bisect {
  public static void main(String... args) throws Exception {
    Path model = Path.of(args[0]);
    boolean useNative = !"java".equals(System.getProperty("diag.kernel", "native"));

    // A fixed pseudo-random token stream. Real text would work too, but a fixed stream keeps the
    // comparison independent of a tokenizer and reproducible across machines.
    int[] tokens = new int[40];
    java.util.Random random = new java.util.Random(20260924L);
    for (int index = 0; index < tokens.length; index++) {
      tokens[index] = 1000 + random.nextInt(20000);
    }

    GgufBatchedMatrixKernel kernel =
        useNative ? RustGgufBatchedMatrixKernel.openBundled() : GgufBatchedMatrixKernel.none();
    try (PureJavaBackend backend = PureJavaBackend.load(model, kernel)) {
      System.out.printf(
          "%n  kernel=%s  quantizedDecode=%s  gatedDeltaNet=%s%n",
          kernel.implementation(),
          System.getProperty("models.native.quantizedDecode", "(default)"),
          System.getProperty("models.native.gatedDeltaNet", "(default)"));
      System.out.printf(
          "%n  %-8s %-14s %-14s %-10s%n", "batch", "max |dlogit|", "mean |dlogit|", "argmax");

      for (int batch : new int[] {1, 2, 3, 4, 7, 8, 9, 16, 32}) {
        float[] batched = viaPrefill(backend, tokens, batch);
        float[] stepped = viaSteps(backend, tokens, batch);
        double worst = 0;
        double total = 0;
        for (int index = 0; index < batched.length; index++) {
          double gap = Math.abs(batched[index] - stepped[index]);
          worst = Math.max(worst, gap);
          total += gap;
        }
        System.out.printf(
            "  %-8d %-14.3e %-14.3e %-10s%n",
            batch,
            worst,
            total / batched.length,
            argmax(batched) == argmax(stepped) ? "same" : "DIFFERS");
      }
      System.out.println();
    }
  }

  /** Logits after the first {@code batch} tokens, read by prefilling all but the last as a batch. */
  private static float[] viaPrefill(PureJavaBackend backend, int[] tokens, int batch) {
    backend.reset();
    try (InferenceSession session = backend.openSession()) {
      if (batch > 1) {
        backend.prefill(session, Arrays.copyOf(tokens, batch - 1), 0);
      }
      return backend.forward(session, tokens[batch - 1], batch - 1).clone();
    }
  }

  /** The same logits, reached one token at a time. */
  private static float[] viaSteps(PureJavaBackend backend, int[] tokens, int batch) {
    backend.reset();
    try (InferenceSession session = backend.openSession()) {
      float[] logits = null;
      for (int index = 0; index < batch; index++) {
        logits = backend.forward(session, tokens[index], index);
      }
      return logits.clone();
    }
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
