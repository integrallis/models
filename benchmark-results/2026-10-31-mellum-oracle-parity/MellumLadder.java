package com.integrallis.models.backend.purejava.llama;

import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.plan.ExecutionPlanner;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Map;

/**
 * Per-layer residual sums for one token, through the shipped kernel, for any Llama-path decoder.
 *
 * <p>Generalises {@code MapleLadderRouted}: it <b>tokenises the prompt itself</b> rather than taking
 * a token id, and prints the token embedding's sum. That matters because the oracle arm
 * ({@code llama-eval-callback -p "<text>"}) chooses its own token from its own vocabulary, and a
 * ladder compared against it at a <i>different</i> token would show a divergence that is really a
 * tokenisation mismatch. The {@code embd} sum is the check: if ours and the oracle's agree, both
 * arms are reducing the same row of the same embedding matrix, and only then does a per-layer
 * comparison mean anything.
 *
 * <p>Prints the route for the same reason {@code MapleLadderRouted} does — passing a kernel and the
 * kernel being eligible are different things, and {@code isEligible} can decline silently.
 */
public final class MellumLadder {
  public static void main(String[] args) throws Exception {
    String prompt = args.length > 1 ? args[1] : "The";
    boolean useNative = args.length > 2 && "native".equals(args[2]);
    GgufBatchedMatrixKernel kernel =
        useNative
            ? com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel.openBundled()
            : GgufBatchedMatrixKernel.none();
    try (Arena arena = Arena.ofConfined()) {
      GgufFile file = GgufParser.parse(Path.of(args[0]), arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(file.metadata());
      int[] tokens = tokenizer.encode(prompt);
      int token = tokens[tokens.length - 1];

      System.out.printf(
          "arch=%s layers=%d slidingWindow=%d local=%s rope(0/3)=%s/%s ropeScale(0/3)=%s/%s%n",
          config.architecture().metadataId(),
          config.numLayers(),
          config.slidingWindow(),
          localCount(config),
          config.usesRope(0),
          config.usesRope(3),
          config.ropeFrequencyScale(0),
          config.ropeFrequencyScale(3));
      System.out.printf(
          "kernel=%s  prompt=%s  tokens=%s  laddering token %d%n",
          kernel.implementation(), prompt, java.util.Arrays.toString(tokens), token);

      KvCache cache = new KvCache(config.numLayers(), 8, config.keyDim(), config.valueDim());
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

  private static String localCount(LlamaConfig config) {
    int local = 0;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (config.usesSlidingWindow(layer)) {
        local++;
      }
    }
    return local + "/" + config.numLayers();
  }
}
