package demo;

import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Every matrix shape a forward pass touches, timed at the batch width a decision uses.
 *
 * <p>A whole forward pass achieves about half the rate of the best single matmul on the same host, so
 * something between the two is slow. The candidates are the shapes that are not the fat FFN ones --
 * a 2560-row output parallelises worse than a 9216-row one -- and the quantisation types that are not
 * Q4_K, since Q5_K and Q6_K decode more per weight.
 *
 * <p>Summing each shape's own measured time over the whole model gives the matmul floor. Whatever the
 * forward pass costs above that is norms, the recurrence, attention and the rest.
 */
public final class PerShape {
  public static void main(String... args) throws Exception {
    int batch = Integer.getInteger("diag.tokens", 18);
    try (Arena arena = Arena.ofShared()) {
      var file = GgufParser.parse(java.nio.file.Path.of(args[0]), arena);

      // Count how many layers carry each shape, so one timing covers the whole model.
      Map<String, Integer> occurrences = new HashMap<>();
      Map<String, String> exemplar = new HashMap<>();
      for (var info : file.tensorInfos()) {
        long[] shape = info.shape();
        if (shape.length != 2 || !info.name().startsWith("blk.")) {
          continue;
        }
        String key = info.name().substring(info.name().indexOf('.', 4) + 1) + " " + info.type();
        occurrences.merge(key, 1, Integer::sum);
        exemplar.putIfAbsent(key, info.name());
      }

      try (RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.openBundled()) {
        System.out.printf("%n  batch %d%n%n", batch);
        System.out.printf(
            "  %-24s %-6s %-6s %-7s %-11s %-11s%n",
            "shape", "type", "count", "rows", "G-MAC/s", "total s");
        double matmulFloor = 0;
        double totalMacs = 0;
        for (var entry : occurrences.entrySet()) {
          var tensor = file.getTensor(exemplar.get(entry.getKey()));
          int cols = (int) tensor.shape()[0];
          int rows = (int) tensor.shape()[1];
          if ((long) rows * cols < 1_000_000L) {
            continue;
          }
          MemorySegment weights = tensor.dataSegment();
          Random random = new Random(5L);
          float[] input = new float[batch * cols];
          for (int index = 0; index < input.length; index++) {
            input[index] = (float) (random.nextGaussian() * 0.35);
          }
          float[] out = new float[batch * rows];
          for (int warm = 0; warm < 8; warm++) {
            kernel.multiply(out, input, weights, tensor.type(), batch, rows, cols);
          }
          double best = Double.MAX_VALUE;
          for (int round = 0; round < 10; round++) {
            long t = System.nanoTime();
            kernel.multiply(out, input, weights, tensor.type(), batch, rows, cols);
            best = Math.min(best, (System.nanoTime() - t) / 1e9);
          }
          double macs = (double) batch * rows * cols;
          double seconds = best * entry.getValue();
          matmulFloor += seconds;
          totalMacs += macs * entry.getValue();
          System.out.printf(
              "  %-24s %-6s %-6d %-7d %-11.1f %-11.4f%n",
              entry.getKey().split(" ")[0],
              tensor.type(),
              entry.getValue(),
              rows,
              macs / best / 1e9,
              seconds);
        }
        System.out.printf(
            "%n  matmul floor for the whole model: %.4f s  (%.1f G-MAC/s aggregate)%n",
            matmulFloor, totalMacs / matmulFloor / 1e9);
        System.out.printf(
            "  a measured forward pass of %d tokens costs about %.3f s;"
                + " anything above the floor is not matmul%n%n",
            batch, 0.357);
      }
    }
  }
}
