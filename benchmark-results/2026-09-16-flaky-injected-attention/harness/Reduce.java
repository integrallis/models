import jdk.incubator.vector.*;
public class Reduce {
  static final VectorSpecies<Float> S = FloatVector.SPECIES_PREFERRED;
  static float reduce(float[] v) { return FloatVector.fromArray(S, v, 0).reduceLanes(VectorOperators.ADD); }
  public static void main(String[] a) {
    float[] v = new float[S.length()];
    java.util.Random r = new java.util.Random(7);
    // find inputs whose sequential sum differs from a pairwise sum
    for (int t = 0; ; t++) {
      for (int i = 0; i < v.length; i++) v[i] = (float) (r.nextGaussian() * Math.pow(10, r.nextInt(8) - 4));
      float seq = 0; for (float x : v) seq += x;
      float cold = reduce(v);
      if (cold == seq) { // cold path is the sequential Java fallback; look for a warm mismatch
        float warm = cold; int changedAt = -1;
        for (int i = 0; i < 200_000; i++) { warm = reduce(v); if (warm != cold) { changedAt = i; break; } }
        if (changedAt >= 0) { System.out.println("lanes=" + v.length + " trial=" + t + " cold(sequential)=" + cold + " warm=" + warm + " firstChangedCall=" + changedAt); return; }
        if (t > 50) { System.out.println("lanes=" + v.length + " no change after warmup; cold=" + cold + " seq=" + seq); return; }
      }
    }
  }
}
