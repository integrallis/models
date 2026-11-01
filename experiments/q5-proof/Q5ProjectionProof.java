package experiments.q5;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.ops.PinnedReduction;
import com.integrallis.vectors.core.GgufStagePlan;
import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.*;

/** Paired projection screening, with exact comparisons before every timed cohort. */
public final class Q5ProjectionProof {
  static volatile float sink;
  static long measure(Runnable task, long budget) {
    long start=System.nanoTime(), calls=0;
    do {task.run();calls++;} while(System.nanoTime()-start<budget);
    return (System.nanoTime()-start)/calls;
  }
  public static void main(String[] args) throws Exception {
    String mode=args[1];int threadCount=Integer.parseInt(args[2]);
    System.setProperty("models.native.kernels.threads", ""+threadCount);
    System.setProperty("vectors.gguf.threads", ""+threadCount);
    try(Arena arena=Arena.ofShared();var nativeKernel=RustGgufBatchedMatrixKernel.open(Path.of(args[0]))) {
      for(int[] shape:new int[][]{{384,384},{1536,384},{384,1536},{128,128}}) for(int batch:new int[]{1,4,8,16,64}) {
        int rows=shape[0],cols=shape[1];var type=mode.equals("q51")?GgufTensorType.Q5_1:GgufTensorType.Q5_0;
        int blockBytes=mode.equals("q51")?24:22;
        var weights=arena.allocate((long)rows*cols/32*blockBytes);
        fill(weights,rows,cols,blockBytes);
        float[] input=new float[batch*cols],output=new float[batch*rows],other=new float[batch*rows],reference=new float[batch*rows],row=new float[cols];
        var random=new SplittableRandom(718329);for(int i=0;i<input.length;i++)input[i]=(float)random.nextDouble(-2,2);
        List<String> names=new ArrayList<>();List<Runnable> arms=new ArrayList<>();
        if(mode.equals("q51")) {
          Runnable serial=()->{for(int r=0;r<rows;r++){GgufTensorValues.dequantizeRow(weights,type,r,cols,row);for(int b=0;b<batch;b++)output[b*rows+r]=PinnedReduction.dot(input,b*cols,row,0,cols);}};
          Runnable parallel=()->TensorOps.ggufExactBatchedMatmul(output,input,weights,type,batch,rows,cols,row);
          names.addAll(List.of("java_serial","java_production","native"));
          arms.add(serial);arms.add(parallel);arms.add(()->nativeKernel.multiply(output,input,weights,type,batch,rows,cols));
          serial.run();System.arraycopy(output,0,reference,0,output.length);
        } else {
          names.addAll(List.of("independent","grouped"));
          arms.add(()->{nativeKernel.multiply(output,input,weights,type,batch,rows,cols);nativeKernel.multiply(other,input,weights,type,batch,rows,cols);});
          arms.add(()->nativeKernel.multiplyDual(output,weights,type,rows,other,weights,type,rows,input,batch,cols));
          arms.get(0).run();System.arraycopy(output,0,reference,0,output.length);
        }
        for(int a=0;a<arms.size();a++){arms.get(a).run();if(!Arrays.equals(output,reference))throw new AssertionError(mode+" parity "+names.get(a)+" first="+Arrays.mismatch(output,reference));if(mode.equals("q50")&&!Arrays.equals(other,reference))throw new AssertionError("second grouped output");measure(arms.get(a),100_000_000L);}
        for(int pair=0;pair<4;pair++)for(int step=0;step<arms.size();step++){
          int a=(step+pair)%arms.size();arms.get(a).run();if(!Arrays.equals(output,reference))throw new AssertionError("warm parity "+names.get(a));
          long ns=measure(arms.get(a),150_000_000L);sink=output[output.length-1];
          System.out.printf(Locale.ROOT,"{\"mode\":\"%s\",\"threads\":%d,\"rows\":%d,\"cols\":%d,\"batch\":%d,\"pair\":%d,\"arm\":\"%s\",\"nsPerCall\":%d}%n",mode,threadCount,rows,cols,batch,pair,names.get(a),ns);
        }
      }
    }
  }
  static void fill(MemorySegment w,int rows,int cols,int size){for(int b=0;b<rows*cols/32;b++){long base=(long)b*size;w.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),base,(short)(0x3000|((b*37)&1023)));int offset=2;if(size==24){w.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),base+2,(short)((b%2==0?0:0x8000)|0x3000|((b*53)&1023)));offset=4;}w.set(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),base+offset,b*0x9e3779b9);for(int n=0;n<16;n++)w.set(ValueLayout.JAVA_BYTE,base+offset+4+n,(byte)(b*13+n*29));}}
}
