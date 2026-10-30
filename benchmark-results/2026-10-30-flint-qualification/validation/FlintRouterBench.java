import com.integrallis.models.backend.purejava.ops.PinnedReduction;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.vectors.core.GgufQ4Kernel;
import java.lang.foreign.*;
import java.util.*;
public class FlintRouterBench {
 static volatile float sink;
 public static void main(String[] args) {
  int rows=256,cols=2048,layers=47;var random=new Random(1729);
  float[] x=new float[cols],out=new float[rows];for(int i=0;i<cols;i++)x[i]=random.nextFloat()-.5f;
  float[][] heap=new float[layers][rows*cols];MemorySegment[] mapped=new MemorySegment[layers];
  try(var arena=Arena.ofConfined()) {
   for(int l=0;l<layers;l++) {for(int i=0;i<heap[l].length;i++)heap[l][i]=random.nextFloat()-.5f;mapped[l]=arena.allocate((long)rows*cols*4,64);MemorySegment.copy(MemorySegment.ofArray(heap[l]),0,mapped[l],0,(long)rows*cols*4);}
   for(int arm=0;arm<2;arm++) {
    for(int iteration=0;iteration<120;iteration++) {
     long start=System.nanoTime();
     for(int l=0;l<layers;l++) {
      if(arm==0)TensorOps.ggufMatmul(out,x,mapped[l],GgufTensorType.F32,rows,cols,null,null,null,null,GgufQ4Kernel.WIDENED);
      else for(int r=0;r<rows;r++)out[r]=PinnedReduction.dot(x,0,heap[l],r*cols,cols);
      sink=out[0];
      if(iteration==0) for(int r=0;r<rows;r++)if(Float.floatToRawIntBits(out[r])!=Float.floatToRawIntBits(PinnedReduction.dot(x,0,heap[l],r*cols,cols)))throw new AssertionError();
     }
     if(iteration>=20)System.out.printf("%d %.6f%n",arm,(System.nanoTime()-start)/1e6);
    }
   }
  }
 }
}
