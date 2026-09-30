package demo;

import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

/**
 * How close is the matrix kernel to what this machine can actually do?
 *
 * <p>A decision is 98.5% the prefill of its own suffix, and that prefill is compute bound. So the
 * only question left about the kernel is whether it is near the ceiling or a long way from it, and
 * that decides whether more kernel work is worth anything at all.
 *
 * <p>Reports achieved multiply-accumulates per second on a real Q4_K and Q6_K weight from the
 * shipped model, at the batch width a decision actually uses. Compare against this core's issue
 * limit: Zen 3 has AVX2 and no AVX-512, and an int8 dot product goes through VPMADDUBSW into
 * VPMADDWD, so the sustained ceiling is roughly 32 int8 multiply-accumulates per cycle per core.
 */
public final class Ceiling {
  public static void main(String... args) throws Exception {
    java.nio.file.Path model = java.nio.file.Path.of(args[0]);
    int cores = Integer.getInteger("diag.cores", Runtime.getRuntime().availableProcessors() / 2);
    double ghz = Double.parseDouble(System.getProperty("diag.ghz", "2.0"));
    double ceiling = cores * ghz * 1e9 * 32;

    try (Arena arena = Arena.ofShared()) {
      var file = GgufParser.parse(model, arena);
      try (RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.openBundled()) {
        System.out.printf(
            "%n  %d physical cores assumed at %.1f GHz, 32 int8 MAC/cycle/core"
                + " -> ceiling %.0f G-MAC/s%n%n",
            cores, ghz, ceiling / 1e9);
        System.out.printf("  %-26s %-8s %-8s %-12s %-10s%n", "tensor", "type", "batch", "G-MAC/s", "of ceiling");
        for (String name :
            new String[] {"blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "output.weight"}) {
          var tensor = file.getTensor(name);
          int cols = (int) tensor.shape()[0];
          int rows = (int) tensor.shape()[1];
          MemorySegment weights = tensor.dataSegment();
          for (int batch : new int[] {1, 18}) {
            Random random = new Random(7L);
            float[] input = new float[batch * cols];
            for (int index = 0; index < input.length; index++) {
              input[index] = (float) (random.nextGaussian() * 0.35);
            }
            float[] out = new float[batch * rows];
            for (int warm = 0; warm < 6; warm++) {
              kernel.multiply(out, input, weights, tensor.type(), batch, rows, cols);
            }
            double best = Double.MAX_VALUE;
            for (int round = 0; round < 8; round++) {
              long t = System.nanoTime();
              kernel.multiply(out, input, weights, tensor.type(), batch, rows, cols);
              best = Math.min(best, (System.nanoTime() - t) / 1e9);
            }
            double macs = (double) batch * rows * cols;
            System.out.printf(
                "  %-26s %-8s %-8d %-12.1f %-10.1f%%%n",
                name, tensor.type(), batch, macs / best / 1e9, 100 * (macs / best) / ceiling);
          }
        }
        System.out.println();
      }
    }
  }
}
