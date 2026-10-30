/* Copyright 2025-2026 Integrallis Software, LLC. SPDX-License-Identifier: Apache-2.0 */
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.rag.*;
import com.integrallis.models.runtime.GenerationLoop;
import java.lang.reflect.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;

/** Remote-only diagnostic. Reflection/timing overhead is included; not qualification evidence. */
public final class FlintKernelTiming {
  public static void main(String[] args) throws Exception {
    var corpus=RagCorpus.loadDefault();
    var test=corpus.cases().stream().filter(c->c.id().equals("auto-glass-deadline")).findFirst().orElseThrow();
    var document=corpus.documents().stream().filter(d->d.id().equals("claims-auto-glass")).findFirst().orElseThrow();
    var prompt=RagPromptRenderer.renderPrompt(test.question(),List.of(new RetrievedDocument(document,10f,1)),RagPromptTemplate.FLINT);
    String promptSha=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(prompt.text().getBytes(StandardCharsets.UTF_8)));
    if(args.length==1 && args[0].equals("--prompt-check")) {System.out.println(promptSha);return;}
    if(args.length!=3 || !System.getProperty("os.name").equals("Linux")) throw new IllegalArgumentException("Remote Linux only: MODEL NATIVE_LIBRARY NEW_REPORT");
    Path report=Path.of(args[2]);if(Files.exists(report))throw new IllegalArgumentException("Report exists");
    System.setProperty("models.purejava.maxContextLength","2048");
    var real=RustGgufBatchedMatrixKernel.open(Path.of(args[1]),16);
    Map<String,long[]> counters=new TreeMap<>();
    boolean[] closed={false};
    var proxy=(GgufBatchedMatrixKernel)Proxy.newProxyInstance(GgufBatchedMatrixKernel.class.getClassLoader(),new Class<?>[]{GgufBatchedMatrixKernel.class},(self,method,arguments)->{
      if(method.getName().equals("close")) {if(closed[0])return null;closed[0]=true;}
      boolean timed=method.getName().startsWith("multiply");
      String key=method.getName();
      if(timed && arguments!=null) {
        for(Object arg:arguments) {
          if(arg instanceof GgufTensorType t)key+="/"+t;
          if(arg instanceof GgufTensorType[] types)key+="/"+Arrays.stream(types).filter(Objects::nonNull).distinct().toList();
        }
        if(method.getName().equals("multiply"))key+="/rows="+arguments[5]+"/cols="+arguments[6];
      }
      long start=timed?System.nanoTime():0;
      try{return method.invoke(real,arguments);}
      catch(InvocationTargetException ex){throw ex.getCause();}
      finally {if(timed){long[] c=counters.computeIfAbsent(key,k->new long[2]);c[0]++;c[1]+=System.nanoTime()-start;}}
    });
    var result=new LinkedHashMap<String,Object>();result.put("kind","instrumented-kernel-boundary-diagnosis");result.put("promptSha256",promptSha);result.put("limitations","Includes reflective/timer overhead. Kernel wall time includes native computation, scheduling, FFM and copies. Outside time includes all other Java work. Not qualification.");
    var runs=new ArrayList<Map<String,Object>>();result.put("runs",runs);
    try(var backend=PureJavaBackend.load(Path.of(args[0]),BackendConfiguration.empty(),proxy)) {
      var loop=new GenerationLoop(backend);var options=SamplingOptions.builder().temperature(0).maxTokens(512).build();
      loop.generate(prompt,options);
      for(int iteration=0;iteration<3;iteration++) {
        counters.clear();long start=System.nanoTime();String text=loop.generate(prompt,options);long total=System.nanoTime()-start;
        long nativeNanos=counters.values().stream().mapToLong(v->v[1]).sum();var items=new LinkedHashMap<String,Object>();
        counters.forEach((k,v)->items.put(k,Map.of("calls",v[0],"millis",v[1]/1e6)));
        var run=new LinkedHashMap<String,Object>();run.put("totalMillis",total/1e6);run.put("kernelMillis",nativeNanos/1e6);run.put("outsideKernelMillis",(total-nativeNanos)/1e6);run.put("rawText",text);run.put("kernels",items);runs.add(run);
      }
    } finally {if(!closed[0])real.close();}
    Files.writeString(report,new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(result),StandardOpenOption.CREATE_NEW);
  }
}
