package com.integrallis.models.backend.purejava.ops;

import com.integrallis.models.backend.purejava.quant.TernaryDequantizer;
import com.sun.management.ThreadMXBean;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.management.ManagementFactory;
import java.nio.ByteOrder;
import java.util.SplittableRandom;
import jdk.incubator.vector.FloatVector;

/** Fixed public-kernel diagnosis; primitive controls preserve the pinned arithmetic order. */
public final class WidthRiskProbe {
  private static final int GROUPS=5, ROWS=64, COLUMNS=64, SOFTMAX=515, PTQ=4096;
  private static volatile float sink;
  private static final ValueLayout.OfShort LE_SHORT=ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final int[] POW3={1,3,9,27,81};
  private final String kernel;
  private final boolean reference;
  private final float[] query=new float[GROUPS*COLUMNS], keys=new float[ROWS*COLUMNS];
  private final float[] scores=new float[GROUPS*ROWS], logits=new float[SOFTMAX], probabilities=new float[SOFTMAX];
  private final byte[] quants=new byte[PTQ];
  private final float[] scales=new float[PTQ/32], floatInput=new float[PTQ], floatSums=new float[8];
  private final Arena arena=Arena.ofConfined();
  private final MemorySegment weights=arena.allocate((PTQ/128)*28L);
  private float result;

  private WidthRiskProbe(String kernel, boolean reference) {
    this.kernel=kernel;this.reference=reference;
    var random=new SplittableRandom(871032);
    for(int i=0;i<query.length;i++)query[i]=(float)random.nextDouble(-1,1);
    for(int i=0;i<keys.length;i++)keys[i]=(float)random.nextDouble(-1,1);
    for(int i=0;i<logits.length;i++)logits[i]=(float)random.nextDouble(-8,8);
    for(int i=0;i<quants.length;i++)quants[i]=(byte)random.nextInt(-127,128);
    for(int i=0;i<scales.length;i++)scales[i]=(float)Math.scalb(1.0,random.nextInt(-8,-2));
    for(long i=0;i<weights.byteSize();i++)weights.set(ValueLayout.JAVA_BYTE,i,(byte)random.nextInt(243));
    for(int block=0;block<PTQ/128;block++)weights.set(LE_SHORT,block*28L+26,Float.floatToFloat16(.125f));
    for(int i=0;i<floatInput.length;i++)floatInput[i]=(float)random.nextDouble(-1,1);
  }
  private void invoke(int iteration) {
    switch(kernel) {
      case "score" -> {
        query[0]=(iteration%31-15)*.03125f;
        if(reference)scoreReference();
        else GroupedQueryAttentionKernel.scoreGroup(query,0,COLUMNS,GROUPS,keys,0,COLUMNS,ROWS,COLUMNS,.125f,scores,0,ROWS);
        result=scores[iteration%scores.length];
      }
      case "softmax" -> {
        System.arraycopy(logits,0,probabilities,0,SOFTMAX);
        probabilities[0]+=(iteration%31-15)*.03125f;
        if(reference)softmaxReference();else GroupedQueryAttentionKernel.softmax(probabilities,0,SOFTMAX);
        result=probabilities[iteration%SOFTMAX];
      }
      case "ptq" -> {
        quants[0]=(byte)(iteration%255-127);
        result=reference?ptqReference():TernaryDequantizer.dotProductPtq1_0Q8_0(weights,0,quants,scales,0,PTQ);
      }
      case "ptqFloat" -> {
        floatInput[0]=(iteration%31-15)*.03125f;
        result=reference?ptqFloatReference():TernaryDequantizer.dotProductPtq1_0(weights,0,floatInput,0,PTQ);
      }
      default -> throw new IllegalArgumentException(kernel);
    }
    sink=result;
  }
  private static float fold(float a0,float a1,float a2,float a3,float a4,float a5,float a6,float a7) {
    return ((a0+a4)+(a2+a6))+((a1+a5)+(a3+a7));
  }
  private void scoreReference() {
    for(int row=0;row<ROWS;row++)for(int head=0;head<GROUPS;head++) {
      float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
      for(int c=0;c<COLUMNS;c+=8) {
        int q=head*COLUMNS+c,k=row*COLUMNS+c;
        a0=Math.fma(query[q],keys[k],a0);a1=Math.fma(query[q+1],keys[k+1],a1);
        a2=Math.fma(query[q+2],keys[k+2],a2);a3=Math.fma(query[q+3],keys[k+3],a3);
        a4=Math.fma(query[q+4],keys[k+4],a4);a5=Math.fma(query[q+5],keys[k+5],a5);
        a6=Math.fma(query[q+6],keys[k+6],a6);a7=Math.fma(query[q+7],keys[k+7],a7);
      }
      scores[head*ROWS+row]=fold(a0,a1,a2,a3,a4,a5,a6,a7)*.125f;
    }
  }
  private void softmaxReference() {
    float maximum=Float.NEGATIVE_INFINITY;
    for(float x:probabilities)maximum=Math.max(maximum,x);
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    int index=0,limit=SOFTMAX&~7;
    for(;index<limit;index+=8) {
      for(int j=0;j<8;j++)probabilities[index+j]=GroupedQueryAttentionKernel.expScalar(probabilities[index+j]-maximum);
      a0+=probabilities[index];a1+=probabilities[index+1];a2+=probabilities[index+2];a3+=probabilities[index+3];
      a4+=probabilities[index+4];a5+=probabilities[index+5];a6+=probabilities[index+6];a7+=probabilities[index+7];
    }
    float sum=fold(a0,a1,a2,a3,a4,a5,a6,a7);
    for(;index<SOFTMAX;index++){float value=GroupedQueryAttentionKernel.expScalar(probabilities[index]-maximum);probabilities[index]=value;sum+=value;}
    float inverse=1f/sum;for(int i=0;i<SOFTMAX;i++)probabilities[i]*=inverse;
  }
  private float ptqReference() {
    float total=0;
    for(int block=0;block<PTQ/128;block++) {
      long base=block*28L;float blockSum=0;
      for(int group=0;group<4;group++) {
        int sum=0;
        for(int j=0;j<32;j++) {
          int index=group*32+j,packed,power;
          if(index<80){packed=index%16;power=index/16;}
          else if(index<120){packed=16+(index-80)%8;power=(index-80)/8;}
          else{packed=24+(index-120)%2;power=(index-120)/2;}
          int q=(Byte.toUnsignedInt(weights.get(ValueLayout.JAVA_BYTE,base+packed))*POW3[power])&255;
          sum+=(((q*3)>>>8)-1)*quants[block*128+index];
        }
        blockSum=Math.fma(sum,scales[block*4+group],blockSum);
      }
      total=Math.fma(blockSum,Float.float16ToFloat(weights.get(LE_SHORT,base+26)),total);
    }
    return total;
  }
  private int trit(long base,int packed,int power) {
    int value=(Byte.toUnsignedInt(weights.get(ValueLayout.JAVA_BYTE,base+packed))*POW3[power])&255;
    return ((value*3)>>>8)-1;
  }
  private float ptqFloatReference() {
    float total=0;
    for(int block=0;block<PTQ/128;block++) {
      long base=block*28L;int at=block*128;
      java.util.Arrays.fill(floatSums,0f);
      for(int n=0;n<5;n++)for(int lane=0;lane<8;lane++) {
        floatSums[lane]=Math.fma(trit(base,lane,n),floatInput[at+n*16+lane],floatSums[lane]);
        floatSums[lane]=Math.fma(trit(base,8+lane,n),floatInput[at+n*16+8+lane],floatSums[lane]);
        floatSums[lane]=Math.fma(trit(base,16+lane,n),floatInput[at+80+n*8+lane],floatSums[lane]);
      }
      float sum=((floatSums[4]+floatSums[0])+(floatSums[6]+floatSums[2]))+((floatSums[5]+floatSums[1])+(floatSums[7]+floatSums[3]));
      for(int n=0;n<4;n++)for(int h=0;h<2;h++)sum=Math.fma(trit(base,24+h,n),floatInput[at+120+n*2+h],sum);
      total=Math.fma(sum,Float.float16ToFloat(weights.get(LE_SHORT,base+26)),total);
    }
    return total;
  }
  private long signature() {
    long value=0x123456789abcdefL;
    for(int iteration=0;iteration<31;iteration++) {
      invoke(iteration);
      float[] output=kernel.equals("score")?scores:kernel.equals("softmax")?probabilities:null;
      if(output!=null)for(float x:output)value=value*31+Float.floatToRawIntBits(x);
      else value=value*31+Float.floatToRawIntBits(result);
    }
    return value;
  }
  public static void main(String[] args) {
    var probe=new WidthRiskProbe(args[0],args[1].equals("reference"));
    var bean=(ThreadMXBean)ManagementFactory.getThreadMXBean();
    if(!bean.isThreadAllocatedMemorySupported())throw new AssertionError("allocation accounting unsupported");
    bean.setThreadAllocatedMemoryEnabled(true);
    long before=bean.getCurrentThreadAllocatedBytes(),start=System.nanoTime();probe.invoke(0);
    long coldNanos=System.nanoTime()-start,coldBytes=bean.getCurrentThreadAllocatedBytes()-before;
    long coldSignature=probe.signature();
    for(int i=0;i<2000;i++)probe.invoke(i);
    System.out.printf("{\"kind\":\"cold\",\"kernel\":\"%s\",\"arm\":\"%s\",\"preferredBits\":%d,\"coldBytes\":%d,\"coldNanos\":%d,\"signature\":\"%016x\"}%n",args[0],args[1],FloatVector.SPECIES_PREFERRED.vectorBitSize(),coldBytes,coldNanos,coldSignature);
    for(int block=0;block<5;block++) {
      before=bean.getCurrentThreadAllocatedBytes();start=System.nanoTime();
      int batch=args[0].equals("ptq")?1020:992;
      long calls=0;
      do {
        for(int i=0;i<batch;i++)probe.invoke((int)calls+i);
        calls+=batch;
      } while(System.nanoTime()-start<250_000_000L);
      long nanos=System.nanoTime()-start,bytes=bean.getCurrentThreadAllocatedBytes()-before;
      System.out.printf("{\"kind\":\"measure\",\"kernel\":\"%s\",\"arm\":\"%s\",\"block\":%d,\"calls\":%d,\"elapsedNanos\":%d,\"nanosPerCall\":%.3f,\"bytesPerCall\":%.3f}%n",args[0],args[1],block,calls,nanos,nanos/(double)calls,bytes/(double)calls);
    }
    long hotSignature=probe.signature();if(hotSignature!=coldSignature)throw new AssertionError("cold/hot output changed");
    System.out.printf("{\"kind\":\"output\",\"signature\":\"%016x\",\"sink\":\"%s\"}%n",hotSignature,Float.toHexString(sink));
    probe.arena.close();
  }
}
