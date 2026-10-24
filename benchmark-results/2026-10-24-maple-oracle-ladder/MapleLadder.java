package com.integrallis.models.backend.purejava.llama;

import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import java.lang.foreign.Arena;
import java.nio.file.Path;

/** Per-layer residual sums for one token, to localise where we leave llama.cpp's ladder. */
public final class MapleLadder {
  public static void main(String[] args) throws Exception {
    int token = Integer.parseInt(args[1]);
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

      LlamaForwardPass.LayerObserver observer =
          (layer, position, state, offset, length) -> {
            double sum = 0.0;
            for (int i = 0; i < length; i++) {
              sum += state[offset + i];
            }
            System.out.printf("l_out-%d %.6f%n", layer, sum);
          };
      LlamaForwardPass pass = new LlamaForwardPass(config, weights, cache, observer);
      pass.forward(token, 0);
    }
  }
}
