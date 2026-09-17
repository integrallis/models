import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.llama.*;
import com.integrallis.models.backend.purejava.cache.KvCache;
import java.lang.reflect.*;
import java.util.*;
public class Drift {
  public static void main(String[] a) throws Exception {
    int iters = Integer.parseInt(a[0]);
    Class<?> tc = Class.forName("com.integrallis.models.backend.purejava.llama.LlamaForwardPassTest");
    Constructor<?> ctor = tc.getDeclaredConstructor(); ctor.setAccessible(true);
    Object t = ctor.newInstance();
    Method build = tc.getDeclaredMethod(a.length > 1 ? a[1] : "buildQ4NanoModel", Random.class); build.setAccessible(true);
    GgufFile file = (GgufFile) build.invoke(t, new Random(42));
    LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
    LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
    int[] tokens = {5, 7, 11, 13, 17, 19, 23, 29};
    Method plan = null;
    for (Method m : tc.getDeclaredMethods()) if (m.getName().equals("executionPlan") && m.getParameterCount()==4) plan = m;
    plan.setAccessible(true);
    Object p = plan.invoke(null, config, weights, tokens.length, false);
    Constructor<?> fp = null;
    for (Constructor<?> c : LlamaForwardPass.class.getDeclaredConstructors()) if (c.getParameterCount()==4 && c.getParameterTypes()[3].getSimpleName().equals("PureJavaExecutionPlan")) fp = c;
    fp.setAccessible(true);
    float[] first = null; int changes = 0; int firstChange = -1; double maxDiff = 0;
    long t0 = System.nanoTime();
    for (int i = 0; i < iters; i++) {
      LlamaForwardPass pass = (LlamaForwardPass) fp.newInstance(config, weights, new KvCache(config.numLayers(), config.contextLength(), config.keyDim(), config.valueDim()), p);
      float[] r = pass.prefill(tokens, 0).clone();
      if (first == null) { first = r; continue; }
      if (!Arrays.equals(first, r)) {
        changes++; if (firstChange < 0) { firstChange = i; System.out.println("FIRST_CHANGE uptimeMs=" + java.lang.management.ManagementFactory.getRuntimeMXBean().getUptime()); }
        for (int k=0;k<r.length;k++) maxDiff = Math.max(maxDiff, Math.abs(r[k]-first[k]));
      }
    }
    System.out.printf("model=%s iters=%d differingFromCold=%d firstDifferingIter=%d maxAbsDiff=%.3g ms=%d%n", a.length>1?a[1]:"buildQ4NanoModel", iters, changes, firstChange, maxDiff, (System.nanoTime()-t0)/1_000_000);
  }
}
