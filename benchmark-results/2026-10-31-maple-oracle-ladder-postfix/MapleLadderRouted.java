package com.integrallis.models.backend.purejava.llama;

import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.plan.ExecutionPlanner;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Map;

/**
 * Per-layer residual sums for one token, routed through the kernel the product actually uses.
 *
 * <p><b>Why this replaces {@code MapleLadder}.</b> That harness calls the observer constructor,
 * and every observer constructor in {@link LlamaForwardPass} hard-codes {@code
 * GgufBatchedMatrixKernel.none()}. So it measured the exact F32 route no matter what was on the
 * classpath, and 168 of maple's tensors are TQ1_0 — the ones whose whole point is that ggml
 * quantizes their activations to Q8_K and the exact route does not. A ladder that cannot reach the
 * TQ1_0 x Q8_K kernel cannot answer whether that kernel closed the gap, which is what #299 asks.
 *
 * <p>The kernel-accepting constructor already existed and is package-private, which is why this
 * class lives in the decoder's package. Nothing in the runtime changed to make this measurable.
 *
 * <p><b>It prints the route rather than assuming it.</b> Passing a kernel is not the same as the
 * kernel being eligible for a given tensor type: {@code isEligible} can decline and the pass falls
 * back silently. The route line below is the evidence that the measured arm is the intended one,
 * and a ladder without it is a number whose provenance cannot be checked.
 */
public final class MapleLadderRouted {
  public static void main(String[] args) throws Exception {
    int token = Integer.parseInt(args[1]);
    boolean useNative = args.length > 2 && "native".equals(args[2]);
    GgufBatchedMatrixKernel kernel =
        useNative
            ? com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel.openBundled()
            : GgufBatchedMatrixKernel.none();
    try (Arena arena = Arena.ofConfined()) {
      GgufFile file = GgufParser.parse(Path.of(args[0]), arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      KvCache cache = new KvCache(config.numLayers(), 8, config.keyDim(), config.valueDim());

      System.out.printf(
          "layers=%d  swa=%s  clamp=%s  rope=%s%n",
          config.numLayers(),
          config.usesSlidingWindow(0) + "/" + config.usesSlidingWindow(3),
          config.swigluClampExp(0),
          config.usesRope(0) + "/" + config.usesRope(3));
      // The route, read rather than inferred. TQ1_0 eligibility is the one that matters here.
      System.out.printf(
          "kernel=%s  tq1_0-supported=%s  tq1_0-eligible-1x2048x2048=%s  halfPrecisionKvCache=%s%n",
          kernel.implementation(),
          kernel.supports(com.integrallis.models.backend.purejava.gguf.GgufTensorType.TQ1_0),
          kernel.isEligible(
              com.integrallis.models.backend.purejava.gguf.GgufTensorType.TQ1_0, 1, 2048, 2048),
          Boolean.getBoolean("models.purejava.halfPrecisionKvCache"));

      LlamaForwardPass.LayerObserver observer =
          (layer, position, state, offset, length) -> {
            double sum = 0.0;
            for (int i = 0; i < length; i++) {
              sum += state[offset + i];
            }
            System.out.printf("l_out-%d %.6f%n", layer, sum);
          };
      LlamaForwardPass pass =
          new LlamaForwardPass(
              config,
              weights,
              cache,
              observer,
              ExecutionPlanner.plan(
                  RuntimeFingerprint.capture(),
                  ModelTopology.from(config.architecture().metadataId(), config, weights),
                  PureJavaPlanConfiguration.fromSystemProperties(Map.of())),
              kernel);
      pass.forward(token, 0);
    }
  }
}
