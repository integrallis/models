package demo;

import com.integrallis.models.api.GroupedDecisionBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.ResumableInferenceBackend;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * A GROUPED ANSWER MUST EQUAL THE SAME QUESTION ASKED ALONE.
 *
 * <p>The toy-model parity test says it does, exactly. On the shipped 4B model it does not: up to
 * 0.10 of probability apart. The difference is not the grouping logic -- it is that a group reads its
 * answer out of a batch of rows while a lone question reads its answer out of a batch of one, and
 * those are different code paths in the native kernel whose results differ by fp32 rounding that the
 * graph then amplifies.
 *
 * <p>Run against both kernels. The Java kernel's batched and single-row paths are bit-identical, so
 * if the diagnosis is right its grouped answers must be exactly its sequential ones, and the native
 * kernel's must not.
 */
public final class Grouped {
  public static void main(String... args) throws Exception {
    Path model = Path.of(args[0]);
    boolean useNative = !"java".equals(System.getProperty("diag.kernel", "native"));
    GgufBatchedMatrixKernel kernel =
        useNative ? RustGgufBatchedMatrixKernel.openBundled() : GgufBatchedMatrixKernel.none();

    int[] evidence = new int[120];
    java.util.Random random = new java.util.Random(11L);
    for (int index = 0; index < evidence.length; index++) {
      evidence[index] = 1000 + random.nextInt(20000);
    }
    int groups = Integer.getInteger("diag.group", 4);
    int[][] suffixes = new int[groups][];
    for (int index = 0; index < groups; index++) {
      int[] suffix = new int[12 + index];
      for (int k = 0; k < suffix.length; k++) {
        suffix[k] = 2000 + random.nextInt(15000);
      }
      suffixes[index] = suffix;
    }

    try (PureJavaBackend backend = PureJavaBackend.load(model, kernel)) {
      System.out.printf(
          "%n  kernel=%s  group=%d  quantizedDecode=%s%n",
          kernel.implementation(), groups, System.getProperty("models.native.quantizedDecode", "(default)"));

      ResumableInferenceBackend resumable = (ResumableInferenceBackend) backend;

      // Warm both paths before either is measured. The grouped path is separate code from the
      // sequential one, so warming only one leaves the other interpreted and the comparison
      // measures compilation rather than arithmetic.
      int warmup = Integer.getInteger("diag.warmup", 6);
      for (int round = 0; round < warmup; round++) {
        backend.reset();
        backend.prefill(evidence, 0);
        ResumableInferenceBackend.Resumption warm = resumable.capture();
        for (int[] suffix : suffixes) {
          resumable.resume(warm);
          if (suffix.length > 1) {
            backend.prefill(Arrays.copyOf(suffix, suffix.length - 1), evidence.length);
          }
          backend.forward(suffix[suffix.length - 1], evidence.length + suffix.length - 1);
        }
        resumable.resume(warm);
        ((GroupedDecisionBackend) backend).decideGrouped(suffixes);
      }
      System.out.printf("  warmup %d rounds of both paths%n", warmup);

      // Sequential: prefill the evidence, resume it per question, prefill the question, read out.
      backend.reset();
      backend.prefill(evidence, 0);
      ResumableInferenceBackend.Resumption point = resumable.capture();
      float[][] alone = new float[groups][];
      for (int index = 0; index < groups; index++) {
        resumable.resume(point);
        int[] suffix = suffixes[index];
        if (suffix.length > 1) {
          backend.prefill(Arrays.copyOf(suffix, suffix.length - 1), evidence.length);
        }
        alone[index] =
            backend
                .forward(suffix[suffix.length - 1], evidence.length + suffix.length - 1)
                .clone();
      }

      resumable.resume(point);
      float[][] grouped = ((GroupedDecisionBackend) backend).decideGrouped(suffixes);

      System.out.printf("%n  %-10s %-14s %-14s %-10s%n", "question", "max |dlogit|", "mean |dlogit|", "argmax");
      for (int index = 0; index < groups; index++) {
        double worst = 0;
        double total = 0;
        for (int k = 0; k < alone[index].length; k++) {
          double gap = Math.abs(alone[index][k] - grouped[index][k]);
          worst = Math.max(worst, gap);
          total += gap;
        }
        System.out.printf(
            "  %-10d %-14.3e %-14.3e %-10s%n",
            index,
            worst,
            total / alone[index].length,
            argmax(alone[index]) == argmax(grouped[index]) ? "same" : "DIFFERS");
      }
      System.out.println();
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
