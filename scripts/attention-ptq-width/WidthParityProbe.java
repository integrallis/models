package com.integrallis.models.backend.purejava.ops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.SplittableRandom;

/** Compares frozen and candidate production classes across physical widths, offsets and tails. */
public final class WidthParityProbe {
  private static final SplittableRandom RANDOM = new SplittableRandom(702151);
  private static long cases;
  private static float[] random(int count) {
    float[] values = new float[count];
    for (int i = 0; i < count; i++) values[i] = (float) RANDOM.nextDouble(-8, 8);
    return values;
  }
  private static void exact(float[] expected, float[] actual) {
    if (expected.length != actual.length) throw new AssertionError("length");
    for (int i = 0; i < expected.length; i++) {
      if (Float.floatToRawIntBits(expected[i]) != Float.floatToRawIntBits(actual[i]))
        throw new AssertionError("case="+cases+" index="+i+" expected="+Float.toHexString(expected[i])+" actual="+Float.toHexString(actual[i]));
    }
    cases++;
  }
  private static URLClassLoader loader(String classes, String libs) throws Exception {
    var paths = new ArrayList<URL>();
    paths.add(Path.of(classes).toUri().toURL());
    try (var files = java.nio.file.Files.list(Path.of(libs))) {
      for (Path file : files.filter(p -> p.toString().endsWith(".jar")).sorted().toList())
        paths.add(file.toUri().toURL());
    }
    return new URLClassLoader(paths.toArray(URL[]::new), ClassLoader.getPlatformClassLoader());
  }
  public static void main(String[] args) throws Exception {
    try (var baseline = loader(args[0], args[2]); var candidate = loader(args[1], args[2])) {
      Class<?> oldAttention = baseline.loadClass("com.integrallis.models.backend.purejava.ops.GroupedQueryAttentionKernel");
      Class<?> newAttention = candidate.loadClass(oldAttention.getName());
      Class<?>[] scoreTypes = {float[].class,int.class,int.class,int.class,float[].class,int.class,int.class,int.class,int.class,float.class,float[].class,int.class,int.class};
      Method oldScore=oldAttention.getMethod("scoreGroup",scoreTypes), newScore=newAttention.getMethod("scoreGroup",scoreTypes);
      Method oldSoftmax=oldAttention.getMethod("softmax",float[].class,int.class,int.class), newSoftmax=newAttention.getMethod("softmax",float[].class,int.class,int.class);
      Class<?>[] ptqTypes={MemorySegment.class,long.class,byte[].class,float[].class,int.class,int.class};
      Method oldPtq=baseline.loadClass("com.integrallis.models.backend.purejava.quant.TernaryDequantizer").getMethod("dotProductPtq1_0Q8_0",ptqTypes);
      Method newPtq=candidate.loadClass("com.integrallis.models.backend.purejava.quant.TernaryDequantizer").getMethod("dotProductPtq1_0Q8_0",ptqTypes);
      // Repeat the complete case set to compare both first-use and compiled production paths.
      for (int repeat=0;repeat<4;repeat++) {
        for(int columns:new int[]{1,3,4,7,8,9,15,16,17,63,64,65,128})
          for(int rows:new int[]{0,1,3,4,5,17}) for(int groups:new int[]{1,3,5}) {
            int offset=repeat%3, stride=columns+5, scoreStride=rows+3;
            float[] q=random(offset+groups*stride),k=random(offset+Math.max(rows,1)*stride);
            float[] expected=random(offset+groups*scoreStride),actual=expected.clone();
            oldScore.invoke(null,q,offset,stride,groups,k,offset,stride,rows,columns,.125f,expected,offset,scoreStride);
            newScore.invoke(null,q,offset,stride,groups,k,offset,stride,rows,columns,.125f,actual,offset,scoreStride);
            exact(expected,actual);
          }
        for(int length:new int[]{1,3,4,7,8,9,15,16,17,31,32,33,257,515,1024})
          for(int offset=0;offset<4;offset++) for(int pattern=0;pattern<4;pattern++) {
            float[] expected=random(offset+length+3);
            for(int i=offset;i<offset+length;i++) expected[i]*=pattern==0?1:pattern==1?16:pattern==2?0:128;
            if(pattern==3 && length>1)expected[offset]=Float.NEGATIVE_INFINITY;
            float[] actual=expected.clone();
            oldSoftmax.invoke(null,expected,offset,length);newSoftmax.invoke(null,actual,offset,length);exact(expected,actual);
          }
        for(int blocks:new int[]{0,1,2,3,17,40}) for(int offset:new int[]{0,32,96}) for(int sourceOffset:new int[]{0,1,13}) {
          byte[] bytes=new byte[sourceOffset+blocks*28],input=new byte[offset+blocks*128];
          for(int i=0;i<bytes.length;i++)bytes[i]=(byte)RANDOM.nextInt(256);
          for(int i=0;i<input.length;i++)input[i]=(byte)RANDOM.nextInt(-128,128);
          var source=MemorySegment.ofArray(bytes);
          float[] scales=random(offset/32+blocks*4);
          for(int block=0;block<blocks;block++)source.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),sourceOffset+block*28L+26,(short)new int[]{0,1,0x3000,0xbc00,0x1400}[block%5]);
          float a=(float)oldPtq.invoke(null,source,(long)sourceOffset,input,scales,offset,blocks*128);
          float b=(float)newPtq.invoke(null,source,(long)sourceOffset,input,scales,offset,blocks*128);
          exact(new float[]{a},new float[]{b});
        }
      }
      System.out.println("{\"cases\":"+cases+",\"exact\":true,\"preferredBits\":"+jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize()+"}");
    }
  }
}
