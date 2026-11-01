/* Copyright 2026 Integrallis Software, LLC. SPDX-License-Identifier: Apache-2.0 */
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.api.InferenceBackend;
import java.nio.file.*;
import com.integrallis.models.backend.purejava.lfm3.*;
import java.util.*;
/** Writes complete logits for independent comparison; never treats fluent output as parity. */
public class Lfm3Logits {
  public static void main(String[] args) throws Exception {
    if(args.length!=4)throw new IllegalArgumentException("MODEL TOKEN_CSV OUTPUT MODE");
    System.setProperty("models.purejava.maxContextLength","2048");
    int[] tokens=Arrays.stream(args[1].split(",")).mapToInt(Integer::parseInt).toArray();
    try(InferenceBackend backend=args[3].startsWith("rust")?RustFfmBackend.load(Path.of(args[0])):PureJavaBackend.load(Path.of(args[0]))){
      String trace=System.getenv("LFM3_TRACE");
      if(trace!=null){
        var df=PureJavaBackend.class.getDeclaredField("decoder");df.setAccessible(true);Object engine=backend;if(backend instanceof RustFfmBackend){var ef=RustFfmBackend.class.getDeclaredField("delegate");ef.setAccessible(true);engine=ef.get(backend);}Object decoder=df.get(engine);
        var pf=decoder.getClass().getDeclaredField("forwardPass");pf.setAccessible(true);Object pass=pf.get(decoder);
        var tf=Lfm3ForwardPass.class.getDeclaredField("trace");tf.setAccessible(true);
        tf.set(pass,(java.util.function.BiConsumer<String,float[]>)(name,data)->{try(var out=new java.io.DataOutputStream(Files.newOutputStream(Path.of(trace,name+".bin")))){for(float v:data)out.writeFloat(v);}catch(Exception e){throw new RuntimeException(e);}});
      }
      long start=System.nanoTime();float[] logits=null;
      if(args[3].equals("serial")){for(int i=0;i<tokens.length;i++)logits=backend.forward(tokens[i],i);}
      else logits=backend.prefill(tokens,0);
      try(var out=new java.io.DataOutputStream(Files.newOutputStream(Path.of(args[2])))){for(float v:logits)out.writeFloat(v);}
      float[] captured=logits;
      var best=java.util.stream.IntStream.range(0,logits.length).boxed().sorted((a,b)->Float.compare(captured[b],captured[a])).limit(10).toList();
      System.out.println("elapsedMs="+(System.nanoTime()-start)/1e6);
      for(int id:best)System.out.println(id+"\t"+captured[id]+"\t"+backend.tokenizer().decode(new int[]{id}));
    }
  }
}
