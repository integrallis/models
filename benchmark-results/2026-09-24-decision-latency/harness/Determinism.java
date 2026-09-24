package demo;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.nio.file.Path;

/**
 * THE SAME CALL, TWICE, MUST GIVE THE SAME BITS.
 *
 * <p>Everything else measured about this model -- calibration, accuracy, the cost of a prompt change
 * -- assumes that. The batch bisection says it does not hold: a single forward of a single token,
 * run twice, differs by 0.33 of a logit. That is not a batching question and not a kernel question,
 * so this asks the narrowest version of it directly.
 *
 * <p>Four shapes, because they fail for different reasons. Twice on one session after a reset, twice
 * on two fresh sessions, twice on the default session, and a repeat of the whole first measurement
 * after other work has run. State that survives a reset, state that survives a session, and a
 * reduction whose order depends on the scheduler each show up in a different one.
 */
public final class Determinism {
  public static void main(String... args) throws Exception {
    Path model = Path.of(args[0]);
    boolean useNative = !"java".equals(System.getProperty("diag.kernel", "native"));
    GgufBatchedMatrixKernel kernel =
        useNative ? RustGgufBatchedMatrixKernel.openBundled() : GgufBatchedMatrixKernel.none();
    int token = 7411;

    int warmup = Integer.getInteger("diag.warmup", 0);
    try (PureJavaBackend backend = PureJavaBackend.load(model, kernel)) {
      System.out.printf("%n  kernel=%s  warmup=%d%n%n", kernel.implementation(), warmup);
      // Warmup is not a nicety here, it is the measurement. Panama vector code that has not been
      // compiled yet runs a different path with a different accumulation order, so an unwarmed
      // comparison measures the JIT and not the kernel. Recorded in this project's working
      // agreement after it once corrupted a whole sweep; walked into again anyway.
      for (int index = 0; index < warmup; index++) {
        freshSession(backend, token + index);
      }

      float[] freshA = freshSession(backend, token);
      float[] freshB = freshSession(backend, token);
      report("two fresh sessions, one token at position 0", freshA, freshB);

      float[] defaultA = defaultSession(backend, token);
      float[] defaultB = defaultSession(backend, token);
      report("default session, reset between", defaultA, defaultB);

      report("fresh session vs default session", freshA, defaultA);

      float[] freshC = freshSession(backend, token);
      report("first fresh session vs the same after other work", freshA, freshC);

      // Ten more, all of which must agree with each other if the answer is a function of the input.
      float[] reference = freshSession(backend, token);
      double worst = 0;
      for (int index = 0; index < 10; index++) {
        float[] again = freshSession(backend, token);
        for (int k = 0; k < reference.length; k++) {
          worst = Math.max(worst, Math.abs(reference[k] - again[k]));
        }
      }
      System.out.printf("  %-52s max %.3e%n", "ten further repeats against each other", worst);
      System.out.println();
    }
  }

  private static float[] freshSession(PureJavaBackend backend, int token) {
    backend.reset();
    try (InferenceSession session = backend.openSession()) {
      return backend.forward(session, token, 0).clone();
    }
  }

  private static float[] defaultSession(PureJavaBackend backend, int token) {
    backend.reset();
    return backend.forward(token, 0).clone();
  }

  private static void report(String label, float[] left, float[] right) {
    double worst = 0;
    int differing = 0;
    for (int index = 0; index < left.length; index++) {
      double gap = Math.abs(left[index] - right[index]);
      worst = Math.max(worst, gap);
      if (gap != 0) {
        differing++;
      }
    }
    System.out.printf(
        "  %-52s max %.3e  differing %d/%d%n", label, worst, differing, left.length);
  }
}
