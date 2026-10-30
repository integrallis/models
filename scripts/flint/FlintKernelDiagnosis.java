/*
 * Copyright 2025-2026 Integrallis Software, LLC
 * SPDX-License-Identifier: Apache-2.0
 */
package com.integrallis.models.bench;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.*;
import java.util.*;

/** Remote diagnostic only: format ablation and identical-input projection comparisons. */
public final class FlintKernelDiagnosis {
  static final Map<String, Stats> stats = new TreeMap<>();
  static final class Stats {
    public int projections, unequalElements, elements;
    public double maximumAbsolute, squaredError, squaredReference;
  }
  public static void main(String[] args) throws Exception {
    if(args.length != 4) throw new IllegalArgumentException("MODEL NATIVE_LIBRARY REFERENCE NEW_REPORT");
    Path report=Path.of(args[3]);
    if(Files.exists(report)) throw new IllegalArgumentException("report already exists");
    System.setProperty("models.purejava.maxContextLength","2048");
    Properties reference = new Properties();
    try(var in=Files.newInputStream(Path.of(args[2]))) { reference.load(in); }
    int[] tokens=ints(reference,"completion.tokens");
    int[] expected=ints(reference,"completion.continuation");
    List<Map<String,Object>> runs=new ArrayList<>();
    List<float[]> javaLogits=new ArrayList<>();
    var mapper=new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    for(String mode:List.of("java", "native-singles", "native-q8", "native-q4", "shadow-java")) {
      long start=System.nanoTime();
      GgufBatchedMatrixKernel kernel=mode.equals("java") ? GgufBatchedMatrixKernel.none() : proxy(Path.of(args[1]), mode);
      try(var backend=PureJavaBackend.load(Path.of(args[0]),kernel)) {
        float[] logits=backend.prefill(tokens,0);
        List<Integer> actual=new ArrayList<>();
        double max=0,error=0,square=0;
        for(int step=0;step<3;step++) {
          int best=0;
          for(int i=0;i<logits.length;i++) {
            if(logits[i]>logits[best])best=i;
            if(!mode.equals("java")) {
              double wanted=javaLogits.get(step)[i], delta=logits[i]-wanted;
              max=Math.max(max,Math.abs(delta));error+=delta*delta;square+=wanted*wanted;
            }
          }
          actual.add(best);
          if(mode.equals("java")) javaLogits.add(logits.clone());
          if(step<2)logits=backend.forward(expected[step],tokens.length+step);
        }
        var row=new LinkedHashMap<String,Object>();row.put("mode",mode);row.put("tokens",actual);
        row.put("referenceTokens",Arrays.copyOf(expected,3));row.put("maximumAbsoluteLogitError",max);
        row.put("relativeRmsLogitError",Math.sqrt(error/Math.max(square,1e-30)));
        row.put("elapsedMillis",(System.nanoTime()-start)/1e6);runs.add(row);
      }
      mapper.writeValue(report.toFile(),Map.of("kind","flint-kernel-diagnosis-v1","runs",runs,"shadowProjectionErrors",stats));
      System.out.println(mode+" completed");
    }
  }
  static GgufBatchedMatrixKernel proxy(Path path,String mode) {
    var nativeKernel=RustGgufBatchedMatrixKernel.open(path,16);
    return (GgufBatchedMatrixKernel) Proxy.newProxyInstance(GgufBatchedMatrixKernel.class.getClassLoader(),new Class<?>[]{GgufBatchedMatrixKernel.class},(proxy,method,a)->{
      String name=method.getName();
      if(mode.equals("native-singles") && (name.contains("Eligible") || name.startsWith("supports")) && !name.equals("isEligible") && !name.equals("supports")) return false;
      if((mode.equals("native-q8") || mode.equals("native-q4")) && (name.contains("Eligible") || name.startsWith("supports")) && a!=null) {
        GgufTensorType allowed=mode.equals("native-q8") ? GgufTensorType.Q8_0 : GgufTensorType.Q4_K;
        for(Object arg:a) {
          if(arg instanceof GgufTensorType type && type!=allowed)return false;
          if(arg instanceof GgufTensorType[] types)for(var type:types)if(type!=allowed)return false;
        }
      }
      Object result;
      try { result=method.invoke(nativeKernel,a); } catch(InvocationTargetException error) { throw error.getCause(); }
      if(mode.equals("shadow-java") && name.startsWith("multiply")) {
        switch(name) {
          case "multiply" -> compare(name,(float[])a[0],(float[])a[1],(MemorySegment)a[2],(GgufTensorType)a[3],(int)a[5],(int)a[6]);
          case "multiplyDual", "multiplyTriple" -> {
            int count=name.equals("multiplyDual")?2:3;
            for(int i=0;i<count;i++)compare(name,(float[])a[4*i],(float[])a[4*count],(MemorySegment)a[4*i+1],(GgufTensorType)a[4*i+2],(int)a[4*i+3],(int)a[4*count+2]);
          }
          case "multiplyGrouped", "multiplyIndependent" -> {
            for(int i=0;i<(int)a[4];i++)compare(name,((float[][])a[0])[i],name.equals("multiplyGrouped")?(float[])a[5]:((float[][])a[5])[i],((MemorySegment[])a[1])[i],((GgufTensorType[])a[2])[i],((int[])a[3])[i],(int)a[7]);
          }
          default -> throw new IllegalStateException("unhandled projection "+name);
        }
      }
      return result;
    });
  }
  static void compare(String operation,float[] actual,float[] input,MemorySegment weights,GgufTensorType type,int rows,int cols) {
    float[] wanted=new float[rows];
    TensorOps.ggufMatmul(wanted,input,weights,type,rows,cols);
    Stats s=stats.computeIfAbsent(operation+":"+type+":"+rows+"x"+cols,k->new Stats());
    s.projections++;s.elements+=rows;
    for(int i=0;i<rows;i++) {
      double delta=actual[i]-wanted[i];
      if(delta!=0)s.unequalElements++;
      s.maximumAbsolute=Math.max(s.maximumAbsolute,Math.abs(delta));s.squaredError+=delta*delta;s.squaredReference+=(double)wanted[i]*wanted[i];
    }
    System.arraycopy(wanted,0,actual,0,rows);
  }
  static int[] ints(Properties p,String key) { return Arrays.stream(p.getProperty(key).split(",")).map(String::trim).mapToInt(Integer::parseInt).toArray(); }
}
