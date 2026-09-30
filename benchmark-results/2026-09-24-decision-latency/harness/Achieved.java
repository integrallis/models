package demo;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Random;

/**
 * What a real forward pass achieves, against what the best single matmul achieves, on one host.
 *
 * <p>A decision is 98.5% the prefill of its own tokens, and that prefill is compute bound. The
 * kernel's inner loop is at its algorithmic ceiling for AVX2 -- Q4_K carries a scale per 32-weight
 * group, so applying it costs a second vector multiply for every one that does useful work, and
 * hoisting the one obviously redundant broadcast changed nothing because the compiler already did.
 *
 * <p>So the question is no longer how fast one matmul goes; it is whether a forward pass, which is
 * about two hundred matmuls of six different shapes plus norms, a recurrence, attention and a
 * vocabulary projection, gets anywhere near that rate. Both numbers have to come from the same host
 * to mean anything, which is what this does.
 *
 * <p>Multiply-accumulates per token is every 2-D weight in the model, since a dense forward pass
 * touches each one once. Measured against a batched prefill, that gives the achieved rate directly.
 */
public final class Achieved {
  public static void main(String... args) throws Exception {
    Path model = Path.of(args[0]);
    int tokens = Integer.getInteger("diag.tokens", 18);

    long macsPerToken = 0;
    try (Arena arena = Arena.ofShared()) {
      var file = GgufParser.parse(model, arena);
      for (var info : file.tensorInfos()) {
        long[] shape = info.shape();
        if (shape.length == 2) {
          macsPerToken += shape[0] * shape[1];
        }
      }
      System.out.printf(
          "%n  every 2-D weight: %.3f G-MAC per token%n", macsPerToken / 1e9);

      // The best single matmul this host can do, at the batch width a decision uses.
      try (RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.openBundled()) {
        double bestRate = 0;
        String bestName = "";
        for (String name : new String[] {"blk.0.ffn_up.weight", "blk.0.ffn_gate.weight"}) {
          var tensor = file.getTensor(name);
          int cols = (int) tensor.shape()[0];
          int rows = (int) tensor.shape()[1];
          MemorySegment weights = tensor.dataSegment();
          Random random = new Random(3L);
          float[] input = new float[tokens * cols];
          for (int index = 0; index < input.length; index++) {
            input[index] = (float) (random.nextGaussian() * 0.35);
          }
          float[] out = new float[tokens * rows];
          for (int warm = 0; warm < 8; warm++) {
            kernel.multiply(out, input, weights, tensor.type(), tokens, rows, cols);
          }
          double best = Double.MAX_VALUE;
          for (int round = 0; round < 10; round++) {
            long t = System.nanoTime();
            kernel.multiply(out, input, weights, tensor.type(), tokens, rows, cols);
            best = Math.min(best, (System.nanoTime() - t) / 1e9);
          }
          double rate = (double) tokens * rows * cols / best;
          if (rate > bestRate) {
            bestRate = rate;
            bestName = name;
          }
        }
        System.out.printf(
            "  best single matmul at batch %d: %.1f G-MAC/s  (%s)%n", tokens, bestRate / 1e9, bestName);

        // What a whole forward pass achieves over the same number of tokens.
        try (PureJavaBackend backend = PureJavaBackend.load(model, kernel)) {
          int[] prompt = new int[tokens];
          Random random = new Random(11L);
          for (int index = 0; index < tokens; index++) {
            prompt[index] = 1000 + random.nextInt(20000);
          }
          for (int warm = 0; warm < 5; warm++) {
            backend.reset();
            try (InferenceSession session = backend.openSession()) {
              backend.prefill(session, prompt, 0);
            }
          }
          double best = Double.MAX_VALUE;
          for (int round = 0; round < 6; round++) {
            backend.reset();
            try (InferenceSession session = backend.openSession()) {
              long t = System.nanoTime();
              backend.prefill(session, prompt, 0);
              best = Math.min(best, (System.nanoTime() - t) / 1e9);
            }
          }
          double achieved = (double) tokens * macsPerToken / best;
          System.out.printf(
              "  whole forward pass, %d tokens: %.3f s, %.1f ms/token, %.1f G-MAC/s%n",
              tokens, best, 1000 * best / tokens, achieved / 1e9);
          System.out.printf(
              "%n  the forward pass runs at %.0f%% of the best single matmul on this host%n%n",
              100 * achieved / bestRate);
        }
      }
    }
  }
}
