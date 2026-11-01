package com.integrallis.models.backend.purejava;

import java.nio.file.Path;
import java.util.Map;

/** Exactly the existing macro's pre-load setup and timed load, without inference. */
public final class Bf16LoadProof {
  public static void main(String[] args) throws Exception {
    Path model = Path.of(args[0]), nativePath = Path.of(args[1]);
    Bf16ModelProof.emit(Map.of("kind", "runtime-origin", "bf16",
        com.integrallis.vectors.core.BFloat16Matrix.class.getProtectionDomain().getCodeSource()
            .getLocation().toString(), "models",
        PureJavaBackend.class.getProtectionDomain().getCodeSource().getLocation().toString()));
    var tasks = Bf16ModelProof.JSON.readTree(Path.of(args[2]).toFile());
    int pair = Integer.parseInt(args[3]);
    if (!tasks.isArray() || tasks.isEmpty()) throw new AssertionError("missing frozen tasks");
    long start = System.nanoTime();
    try (var backend = Bf16ModelProof.load(model, nativePath, 0)) {
      long elapsed = System.nanoTime() - start;
      Bf16ModelProof.emit(Map.of("kind", "load", "nanos", elapsed,
          "diagnostics", backend.diagnostics(), "phase", "load-only", "pair", pair));
    }
  }
}
