import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.security.MessageDigest;
import java.nio.ByteBuffer;
import java.util.Random;

/** Times TensorOps.swiGlu and digests its output, so serial and parallel runs can be compared. */
public final class SwiGluParallelProbe {
  public static void main(String[] args) throws Exception {
    int size = 14336, layers = 32, reps = 25;
    Random random = new Random(20261101L);
    float[] gate = new float[size];
    float[] up = new float[size];
    for (int i = 0; i < size; i++) {
      gate[i] = (float) (random.nextGaussian() * 2.0);
      up[i] = (float) (random.nextGaussian() * 2.0);
    }
    float[] out = new float[size];
    double best = Double.MAX_VALUE;
    for (int r = 0; r < reps; r++) {
      long t0 = System.nanoTime();
      for (int l = 0; l < layers; l++) TensorOps.swiGlu(out, 0, gate, 0, up, 0, size);
      best = Math.min(best, (System.nanoTime() - t0) / 1e6);
    }
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    ByteBuffer buffer = ByteBuffer.allocate(size * 4);
    for (float v : out) buffer.putInt(Float.floatToRawIntBits(v));
    byte[] hash = digest.digest(buffer.array());
    StringBuilder hex = new StringBuilder();
    for (int i = 0; i < 8; i++) hex.append(String.format("%02x", hash[i]));
    System.out.printf("threshold=%-9s  %7.3f ms/token   out-sha256[0:8]=%s%n",
        System.getProperty("models.purejava.swiGluParallelThreshold", "default(4096)"), best, hex);
  }
}
