package com.integrallis.models.backend.onnx;
public class Cold {
  public static void main(String[] a) {
    int n = 216 * 480; float[] xs = new float[n]; float[] out = new float[n];
    java.util.Random r = new java.util.Random(1); for (int i = 0; i < n; i++) xs[i] = (float) r.nextGaussian() * 2f;
    Tensor x = Tensor.ofFloats(new long[] {1, 216, 480}, xs);
    long total = 0;
    for (int call = 0; call < 255; call++) {
      long t = System.nanoTime(); Elementwise.erf(x); long dt = System.nanoTime() - t; total += dt;
      if (call < 6 || call % 50 == 0) System.out.printf("call %3d %8.1f us%n", call, dt / 1e3);
    }
    System.out.printf("255 calls total %.1f ms%n", total / 1e6);
  }
}
