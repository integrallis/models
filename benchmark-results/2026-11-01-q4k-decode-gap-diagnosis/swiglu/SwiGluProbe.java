import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Random;

/**
 * Prices the swiGlu routing gate. LlamaForwardPass.usesVectorizedSwiGlu() is true only for QWEN2 /
 * QWEN3, so every llama-architecture model runs the scalar TensorOps.swiGlu with one Math.exp per
 * element. This measures what the Panama route would buy, and by how much the two disagree --
 * because widening the gate changes llama-arch model outputs, which is a re-qualification question,
 * not a free win.
 */
public final class SwiGluProbe {
  public static void main(String[] args) {
    int size = 14336;           // mathstral-7B FFN hidden dim
    int layers = 32;            // one token's worth of FFN activations
    int reps = 20;
    Random random = new Random(20261101L);
    float[] gate = new float[size];
    float[] up = new float[size];
    for (int i = 0; i < size; i++) {
      gate[i] = (float) (random.nextGaussian() * 2.0);
      up[i] = (float) (random.nextGaussian() * 2.0);
    }
    float[] outScalar = new float[size];
    float[] outVector = new float[size];

    // correctness/divergence
    TensorOps.swiGlu(outScalar, 0, gate, 0, up, 0, size);
    VectorUtil.swiGlu(outVector, 0, gate, 0, up, 0, size);
    double maxAbs = 0, maxRel = 0;
    int differing = 0;
    for (int i = 0; i < size; i++) {
      double diff = Math.abs((double) outScalar[i] - outVector[i]);
      if (outScalar[i] != outVector[i]) differing++;
      double rel = diff / (Math.abs((double) outScalar[i]) + 1e-9);
      maxAbs = Math.max(maxAbs, diff);
      maxRel = Math.max(maxRel, rel);
    }

    double bestScalar = Double.MAX_VALUE, bestVector = Double.MAX_VALUE;
    for (int r = 0; r < reps; r++) {
      long t0 = System.nanoTime();
      for (int l = 0; l < layers; l++) TensorOps.swiGlu(outScalar, 0, gate, 0, up, 0, size);
      double ts = (System.nanoTime() - t0) / 1e6;

      long t1 = System.nanoTime();
      for (int l = 0; l < layers; l++) VectorUtil.swiGlu(outVector, 0, gate, 0, up, 0, size);
      double tv = (System.nanoTime() - t1) / 1e6;

      bestScalar = Math.min(bestScalar, ts);
      bestVector = Math.min(bestVector, tv);
    }
    System.out.printf("one token's FFN activations (%d x %d elements):%n", layers, size);
    System.out.printf("  TensorOps.swiGlu (scalar Math.exp) : %.3f ms/token%n", bestScalar);
    System.out.printf("  VectorUtil.swiGlu (Panama)         : %.3f ms/token  -> %.2fx faster%n",
        bestVector, bestScalar / bestVector);
    System.out.printf("  divergence: %d/%d lanes differ, max abs %.3e, max rel %.3e%n",
        differing, size, maxAbs, maxRel);
    System.out.printf("  checksum %f / %f%n", outScalar[0], outVector[0]);
  }
}
