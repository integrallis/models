import com.integrallis.vectors.core.*;
import java.util.*;
import java.lang.foreign.*;
import java.lang.invoke.*;
public class GlinerSgemmScreen {
 public static void main(String[] args) throws Exception {
  var arena=Arena.ofShared();
  var symbol=SymbolLookup.libraryLookup(args[0],arena).find("screen_sgemm").orElseThrow();
  var kernel=Linker.nativeLinker().downcallHandle(symbol,FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,ValueLayout.ADDRESS,ValueLayout.ADDRESS,ValueLayout.JAVA_LONG,ValueLayout.JAVA_LONG,ValueLayout.JAVA_LONG));
  System.out.println("Synthetic matrices only; no model weights. Java="+System.getProperty("java.version")+" CPUs="+Runtime.getRuntime().availableProcessors());
  for(int[] shape:new int[][]{{1024,1024},{4096,1024},{1024,4096}}) for(int batch:new int[]{32,128,512}) {
   int rows=shape[0],cols=shape[1];var random=new Random(20261101);
   float[] weights=new float[rows*cols],input=new float[batch*cols],a=new float[batch*rows],b=a.clone();
   for(int i=0;i<weights.length;i++)weights[i]=random.nextFloat()-.5f;
   for(int i=0;i<input.length;i++)input[i]=random.nextFloat()-.5f;
   var nw=arena.allocateFrom(ValueLayout.JAVA_FLOAT,weights);
   var nx=arena.allocateFrom(ValueLayout.JAVA_FLOAT,input);
   var ny=arena.allocate((long)b.length*4,64);
   Runnable current=()-> {
    Thread[] threads=new Thread[8];
    for(int w=0;w<8;w++){final int worker=w;threads[w]=Thread.ofVirtual().start(()->{
     for(int row=worker;row<batch;row+=8)VectorUtil.batchDotProductExact(input,row*cols,weights,0,cols,rows,cols,a,row*rows);
    });}
    for(Thread t:threads)try{t.join();}catch(InterruptedException e){throw new RuntimeException(e);}
   };
   Runnable candidate=()->{try{kernel.invokeExact(nx,nw,ny,(long)batch,(long)cols,(long)rows);}catch(Throwable t){throw new RuntimeException(t);}};
   for(int i=0;i<3;i++){current.run();candidate.run();}
   double baseline=0,improved=0;
   for(int i=0;i<8;i++){
    long start=System.nanoTime();current.run();baseline+=System.nanoTime()-start;
    start=System.nanoTime();candidate.run();improved+=System.nanoTime()-start;
   }
   b=ny.toArray(ValueLayout.JAVA_FLOAT);
   double maxError=0;for(int i=0;i<a.length;i++){
    maxError=Math.max(maxError,Math.abs(a[i]-b[i]));
    if(Math.abs(a[i]-b[i])>1e-4+1e-4*Math.abs(a[i]))throw new AssertionError("parity "+i);
   }
   System.out.printf(Locale.ROOT,"rows=%d cols=%d batch=%d currentMs=%.3f rustSgemmMs=%.3f speedup=%.3f maxError=%.8f%n",rows,cols,batch,baseline/8e6,improved/8e6,baseline/improved,maxError);
  }
 }
}
