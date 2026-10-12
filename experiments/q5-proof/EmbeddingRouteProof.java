package experiments.q5;
import com.fasterxml.jackson.databind.*;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.*;
import com.integrallis.models.backend.purejava.gguf.*;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.ops.*;
import com.integrallis.vectors.core.GgufStagePlan;
import java.lang.foreign.*;
import java.nio.file.*;
import java.util.*;

/** Public embedding API qualification; route counters are kept outside timed calls. */
public final class EmbeddingRouteProof {
 static final ObjectMapper JSON=new ObjectMapper();
 static final com.sun.management.ThreadMXBean ALLOC=(com.sun.management.ThreadMXBean)java.lang.management.ManagementFactory.getThreadMXBean();
 static volatile float sink;
 static class Parallel implements GgufBatchedMatrixKernel {
  record Key(float[] output,float[] input,MemorySegment weight,int batch,int rows,int cols){}
  final Map<Key,GgufStagePlan> plans=new HashMap<>();
  final ThreadLocal<float[]> scratch=ThreadLocal.withInitial(()->new float[0]);
  public boolean supports(GgufTensorType type){return type==GgufTensorType.Q5_1;}
  public void multiply(float[] out,float[] in,MemorySegment w,GgufTensorType type,int batch,int rows,int cols){
   plans.computeIfAbsent(new Key(out,in,w,batch,rows,cols),k->GgufStagePlan.of(GgufStagePlan.stage(rows,(from,to)->{
    float[] row=scratch.get();if(row.length<cols){row=new float[cols];scratch.set(row);}
    for(int r=from;r<to;r++){GgufTensorValues.dequantizeRow(w,type,r,cols,row);for(int b=0;b<batch;b++)out[b*rows+r]=PinnedReduction.dot(in,b*cols,row,0,cols);}
   }))).execute();
  }
 }
 static class Observed implements GgufBatchedMatrixKernel {
  final GgufBatchedMatrixKernel delegate;long calls;final Map<String,Long> shapes=new TreeMap<>();boolean observe;
  Observed(GgufBatchedMatrixKernel d){delegate=d;}
  public boolean supports(GgufTensorType t){return delegate.supports(t);}
  public boolean isEligible(GgufTensorType t,int b,int r,int c){return delegate.isEligible(t,b,r,c);}
  public void multiply(float[] out,float[] in,MemorySegment w,GgufTensorType t,int b,int r,int c){if(observe){calls++;shapes.merge(t+":"+b+":"+r+":"+c,1L,Long::sum);}delegate.multiply(out,in,w,t,b,r,c);}
  public void close(){delegate.close();}
 }
 static void emit(Object x)throws Exception{System.out.println(JSON.writeValueAsString(x));}
 public static void main(String[] args)throws Exception{
  String arm=args[0];Path model=Path.of(args[1]),nativePath=Path.of(args[2]);JsonNode cases=JSON.readTree(Path.of(args[3]).toFile());String mode=args[4];int repetitions=Integer.parseInt(args[5]);
  ALLOC.setThreadAllocatedMemoryEnabled(true);
  GgufBatchedMatrixKernel provider=switch(arm){case "native"->RustGgufBatchedMatrixKernel.open(nativePath);case "parallel"->new Parallel();default->GgufBatchedMatrixKernel.none();};
  Observed observed=new Observed(provider);long start=System.nanoTime();
  try(var backend=PureJavaBackend.load(model,observed);var embedding=GgufEmbeddingBackend.builder(backend).normalize(true).build()){
   emit(Map.of("kind","load","arm",arm,"nanos",System.nanoTime()-start));
   List<String> texts=new ArrayList<>();List<Integer> original=new ArrayList<>();List<Integer> counts=new ArrayList<>();
   for(JsonNode item:cases){String text=item.path("text").asText();int before=backend.tokenizer().encode(text).length;original.add(before);while(backend.tokenizer().encode(text).length>512){int last=text.lastIndexOf(' ');if(last<0)throw new AssertionError("cannot truncate token limit");text=text.substring(0,last);}texts.add(text);counts.add(backend.tokenizer().encode(text).length);}
   observed.observe=true;float[] first=embedding.embed(texts.getFirst());observed.observe=false;
   emit(Map.of("kind","route","calls",observed.calls,"shapes",observed.shapes,"first",first));
   if(!arm.equals("java")&&observed.calls==0)throw new AssertionError("injected route did not execute");
   if(mode.equals("timing")){for(int n=0;n<10;n++)embedding.embed(texts.get(n%texts.size()));}
   for(int repeat=0;repeat<repetitions;repeat++)for(int n=0;n<texts.size();n++){
    JsonNode item=cases.get(n);long allocated=ALLOC.getTotalThreadAllocatedBytes();long begin=System.nanoTime();float[] values=embedding.embed(texts.get(n));long elapsed=System.nanoTime()-begin;allocated=ALLOC.getTotalThreadAllocatedBytes()-allocated;sink=values[0];
    Map<String,Object> row=new LinkedHashMap<>();row.put("kind","embedding");row.put("id",item.path("id").asText());row.put("dataset",item.path("dataset").asText());row.put("textKind",item.path("kind").asText());row.put("repeat",repeat);row.put("originalTokens",original.get(n));row.put("tokens",counts.get(n));row.put("nanos",elapsed);row.put("allocatedBytes",allocated);row.put("values",values);emit(row);
   }
   emit(Map.of("kind","completed","arm",arm,"cases",texts.size(),"repeats",repetitions));
  }
 }
}
