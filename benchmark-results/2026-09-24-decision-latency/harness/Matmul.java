package demo;

import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

/**
 * ONE MATMUL, TWO KERNELS, THE SAME NUMBERS EXPECTED.
 *
 * <p>The whole-model bisection said the native kernel's answer changes the moment a prefill batch is
 * two rows or more, while the Java kernel is bit-identical from one row to thirty-two. That points
 * at one projection rather than at the graph, so this takes a real Q4_K tensor out of the shipped
 * model, feeds it the same activations at several batch sizes, and compares:
 *
 * <ul>
 *   <li>native batch N against Java batch N -- do the two implementations agree at all?
 *   <li>native batch N against native row-at-a-time -- is the native kernel consistent with itself?
 *   <li>native batch N twice -- is it even deterministic?
 * </ul>
 *
 * <p>Java batch N against Java row-at-a-time is the control: it should be exactly zero.
 */
public final class Matmul {
  public static void main(String... args) throws Exception {
    java.nio.file.Path model = java.nio.file.Path.of(args[0]);
    try (Arena arena = Arena.ofShared()) {
      var file = GgufParser.parse(model, arena);
      var tensor = file.getTensor(args.length > 1 ? args[1] : "blk.0.ffn_down.weight");
      GgufTensorType type = tensor.type();
      int cols = (int) tensor.shape()[0];
      int rows = (int) tensor.shape()[1];
      MemorySegment weights = tensor.dataSegment();
      System.out.printf("%n  tensor %s  type %s  rows %d  cols %d%n", tensor.name(), type, rows, cols);

      try (RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.openBundled()) {
        System.out.printf("  kernel %s%n%n", kernel.implementation());
        System.out.printf(
            "  %-6s %-13s %-13s %-13s %-13s%n",
            "batch", "rust vs java", "rust self", "rust rerun", "java self");

        for (int batch : new int[] {1, 2, 3, 4, 8, 16}) {
          Random random = new Random(4242L + batch);
          float[] input = new float[batch * cols];
          for (int index = 0; index < input.length; index++) {
            input[index] = (float) (random.nextGaussian() * 0.35);
          }
          // Real transformer activations are not Gaussian: a handful of channels carry values
          // orders of magnitude larger than the rest, and those outliers are what set the scale an
          // activation-quantizing kernel must cover. A Gaussian input understates the error every
          // such kernel makes, so the outliers are put back deliberately.
          double outlier = Double.parseDouble(System.getProperty("diag.outlier", "0"));
          if (outlier > 0) {
            for (int b = 0; b < batch; b++) {
              for (int k = 0; k < 8; k++) {
                input[b * cols + (k * 37 + b) % cols] = (float) (outlier * (k % 2 == 0 ? 1 : -1));
              }
            }
          }

          // Warm every path that is about to be compared. Panama and the FFM downcall stubs both
          // run a different, slower, differently-rounded path before they are compiled, so an
          // unwarmed comparison measures the JIT. This project's working agreement records that;
          // the first version of this harness ignored it and reported a kernel bug that was not one.
          int warmup = Integer.getInteger("diag.warmup", 8);
          float[] discard = new float[batch * rows];
          float[] discardRow = new float[rows];
          for (int w = 0; w < warmup; w++) {
            kernel.multiply(discard, input, weights, type, batch, rows, cols);
            kernel.multiply(discardRow, input, weights, type, 1, rows, cols);
            TensorOps.ggufBatchedMatmul(
                discard, input, weights, type, batch, rows, cols,
                new byte[batch * cols], new float[batch * ((cols + 31) / 32)],
                new int[batch * ((cols + 3) / 4)], new short[batch * ((cols + 15) / 16)],
                new float[batch * rows * 8],
                com.integrallis.vectors.core.GgufQ4Kernel.UNSIGNED_PAIRWISE,
                com.integrallis.vectors.core.GgufQ6BatchedKernel.TWO_QUERY_BLOCK);
            TensorOps.ggufMatmul(discardRow, input, weights, type, rows, cols);
          }

          float[] rust = new float[batch * rows];
          kernel.multiply(rust, input, weights, type, batch, rows, cols);

          float[] rustAgain = new float[batch * rows];
          kernel.multiply(rustAgain, input, weights, type, batch, rows, cols);

          float[] rustRows = new float[batch * rows];
          float[] rowInput = new float[cols];
          float[] rowOutput = new float[rows];
          for (int b = 0; b < batch; b++) {
            System.arraycopy(input, b * cols, rowInput, 0, cols);
            kernel.multiply(rowOutput, rowInput, weights, type, 1, rows, cols);
            System.arraycopy(rowOutput, 0, rustRows, b * rows, rows);
          }

          float[] java = new float[batch * rows];
          TensorOps.ggufBatchedMatmul(
              java,
              input,
              weights,
              type,
              batch,
              rows,
              cols,
              new byte[batch * cols],
              new float[batch * ((cols + 31) / 32)],
              new int[batch * ((cols + 3) / 4)],
              new short[batch * ((cols + 15) / 16)],
              new float[batch * rows * 8],
              com.integrallis.vectors.core.GgufQ4Kernel.UNSIGNED_PAIRWISE,
              com.integrallis.vectors.core.GgufQ6BatchedKernel.TWO_QUERY_BLOCK);

          float[] javaRows = new float[batch * rows];
          for (int b = 0; b < batch; b++) {
            System.arraycopy(input, b * cols, rowInput, 0, cols);
            TensorOps.ggufMatmul(rowOutput, rowInput, weights, type, rows, cols);
            System.arraycopy(rowOutput, 0, javaRows, b * rows, rows);
          }

          System.out.printf(
              "  %-6d %-13.3e %-13.3e %-13.3e %-13.3e%n",
              batch,
              relative(rust, java),
              relative(rust, rustRows),
              relative(rust, rustAgain),
              relative(java, javaRows));
        }
        System.out.println();
      }
    }
  }

  /** Largest absolute difference scaled by the magnitude of the reference. */
  private static double relative(float[] actual, float[] expected) {
    double worst = 0;
    double scale = 0;
    for (int index = 0; index < expected.length; index++) {
      worst = Math.max(worst, Math.abs(actual[index] - expected[index]));
      scale = Math.max(scale, Math.abs(expected[index]));
    }
    return scale == 0 ? worst : worst / scale;
  }
}
