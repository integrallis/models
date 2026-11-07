package com.integrallis.models.experiments;

import com.integrallis.models.backend.purejava.llama.*;
import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.plan.PureJavaExecutionPlan;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.vision.ClipImagePreprocessor;
import java.awt.image.BufferedImage;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;

/** Synthetic refactor regression probe, reusing the repository's deterministic test fixtures. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Fork(jvmArgsPrepend={"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED", "-Xmx512m"})
public class RefactorBenchmark {
  @State(Scope.Thread)
  public static class Decoder {
    @Param({"buildNanoModel", "buildQ4KNanoModel"}) public String fixture;
    LlamaForwardPass pass;
    int[] tokens;
    float[] embeddings;
    @Setup(Level.Trial)
    public void setup() throws Exception {
      Class<?> fixtures=Class.forName("com.integrallis.models.backend.purejava.llama.LlamaForwardPassTest");
      Method build=fixtures.getDeclaredMethod(fixture,Random.class);
      build.setAccessible(true);
      var constructor=fixtures.getDeclaredConstructor(); constructor.setAccessible(true);
      GgufFile file=(GgufFile)build.invoke(constructor.newInstance(),new Random(42));
      var config=LlamaConfig.fromMetadata(file.metadata());
      var weights=LlamaWeights.fromGgufFile(file,config);
      Method plan=fixtures.getDeclaredMethod("executionPlan",LlamaConfig.class,LlamaWeights.class,int.class,boolean.class);
      plan.setAccessible(true);
      var executionPlan=(PureJavaExecutionPlan)plan.invoke(null,config,weights,4,false);
      pass=new LlamaForwardPass(config,weights,new KvCache(config.numLayers(),config.contextLength(),config.keyDim(),config.valueDim()),executionPlan);
      tokens=new int[16];
      embeddings=new float[tokens.length*config.embeddingDim()];
      float[] row=new float[config.embeddingDim()];
      for(int i=0;i<tokens.length;i++) {
        tokens[i]=(i+1)%config.vocabSize();
        weights.embedToken(tokens[i],row);
        System.arraycopy(row,0,embeddings,i*row.length,row.length);
      }
    }
  }
  @State(Scope.Thread)
  public static class Image {
    BufferedImage image;
    float[] mean={.48145466f,.4578275f,.40821073f};
    float[] std={.26862954f,.26130258f,.27577711f};
    @Setup public void setup() {
      image=new BufferedImage(127,83,BufferedImage.TYPE_INT_RGB);
      Random r=new Random(42);
      for(int y=0;y<83;y++)for(int x=0;x<127;x++)image.setRGB(x,y,r.nextInt());
    }
  }
  @Benchmark public float[] tokenPrefillAndDecode(Decoder d) {
    d.pass.reset(); d.pass.prefill(d.tokens,0); return d.pass.forward(3,d.tokens.length);
  }
  @Benchmark public float[] embeddingPrefillAndDecode(Decoder d) {
    d.pass.reset(); d.pass.prefillEmbeddings(d.embeddings,0); return d.pass.forward(3,d.tokens.length);
  }
  @Benchmark public float[] imagePreprocess(Image image) {
    return ClipImagePreprocessor.preprocess(image.image,96,image.mean,image.std);
  }
  // Separate differential check hashes every float, including intermediate logits, in both arms.
  public static void main(String[] args) throws Exception {
    var benchmark=new RefactorBenchmark();
    for(String fixture:new String[]{"buildNanoModel","buildQ4KNanoModel"}) {
      Decoder d=new Decoder(); d.fixture=fixture; d.setup();
      for (String flag : new String[]{"usesBatchedPrefill", "usesGroupedBatchedPrefill", "usesFinalLayerPrefillPruning", "usesFinalLayerKvOnlyPrefill"}) {
        Method mode=LlamaForwardPass.class.getDeclaredMethod(flag); mode.setAccessible(true);
        System.out.println("ROUTE "+fixture+" "+flag+"="+mode.invoke(d.pass));
      }
      d.pass.reset(); print(fixture+"-token-prefill",d.pass.prefill(d.tokens,0));
      for(int i=0;i<8;i++) print(fixture+"-token-decode-"+i,d.pass.forward((i+3)%32,16+i));
      d.pass.reset(); print(fixture+"-image-prefill",d.pass.prefillEmbeddings(d.embeddings,0));
      for(int i=0;i<8;i++) print(fixture+"-image-decode-"+i,d.pass.forward((i+3)%32,16+i));
      print(fixture+"-repeat",benchmark.embeddingPrefillAndDecode(d));
    }
    Image image=new Image(); image.setup();
    print("pixels",benchmark.imagePreprocess(image));
  }
  static void print(String name,float[] values) throws Exception {
    ByteBuffer buffer=ByteBuffer.allocate(values.length*Float.BYTES);
    for(float v:values) {
      if(!Float.isFinite(v))throw new IllegalStateException("Nonfinite output");
      buffer.putInt(Float.floatToRawIntBits(v));
    }
    System.out.println(name+"\t"+values.length+"\t"+HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(buffer.array())));
  }
}
