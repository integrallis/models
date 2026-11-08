import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import static java.lang.foreign.ValueLayout.*;

/** External attention substitution for component localization, never a shipping backend. */
final class OracleAttention implements BatchedCausalAttentionKernel {
  private final Arena arena = Arena.ofConfined();
  private final MethodHandle invoke;
  private final Map<Integer, Layer> layers = new HashMap<>();
  private int calls;
  private record Layer(MemorySegment keys, MemorySegment values, int heads, int width, int capacity,
                       int[] positions) {}

  public OracleAttention() {
    if (!System.getProperty("os.name").equals("Linux")) throw new IllegalStateException("Remote Linux only");
    var symbols = SymbolLookup.libraryLookup(Path.of(System.getProperty("vision.oracle.attention.library")), arena);
    invoke = Linker.nativeLinker().downcallHandle(symbols.find("oracle_attention").orElseThrow(),
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
  }

  public boolean isEligible(int layer, int start, int batch, int heads, int kvHeads,
                            int keyLength, int valueLength, int capacity, int window) {
    if (layer < 0 || start < 0 || batch < 1 || heads < 1 || heads != kvHeads ||
        keyLength < 1 || keyLength != valueLength || capacity < batch || start > capacity-batch || window != 0)
      throw new IllegalArgumentException("Unsupported diagnostic attention; fallback is forbidden");
    var state = layers.get(layer);
    if (state == null ? start != 0 : state.positions()[0] != start || state.heads() != heads ||
        state.width() != keyLength || state.capacity() != capacity)
      throw new IllegalArgumentException("Discontinuous diagnostic attention");
    return true;
  }

  public void attend(float[] output, float[] query, float[] key, float[] value, int layer,
      int start, int batch, int heads, int kvHeads, int width, int valueLength, int capacity, int window) {
    isEligible(layer, start, batch, heads, kvHeads, width, valueLength, capacity, window);
    int dimension = Math.multiplyExact(heads, width), count = Math.multiplyExact(batch, dimension);
    if (output.length < count || query.length < count || key.length < count || value.length < count)
      throw new IllegalArgumentException("Short diagnostic buffer");
    var state = layers.computeIfAbsent(layer, ignored -> {
      long bytes = Math.multiplyExact((long) capacity * dimension, Float.BYTES);
      return new Layer(arena.allocate(bytes, 64), arena.allocate(bytes, 64), heads, width, capacity, new int[1]);
    });
    for (int row=0; row<batch; row++) for (int head=0; head<heads; head++) {
      long destination = ((long) head*capacity+start+row)*width*4;
      int source = row*dimension+head*width;
      MemorySegment.copy(key, source, state.keys(), JAVA_FLOAT, destination, width);
      MemorySegment.copy(value, source, state.values(), JAVA_FLOAT, destination, width);
    }
    try (var scratch = Arena.ofConfined()) {
      var q = scratch.allocate((long) count*4, 64);
      var y = scratch.allocate((long) count*4, 64);
      MemorySegment.copy(query, 0, q, JAVA_FLOAT, 0, count);
      int result = (int) invoke.invokeExact(q, state.keys(), state.values(), y,
          batch, start, heads, width, capacity, 8);
      if (result != 0) throw new IllegalStateException("Oracle attention status " + result);
      MemorySegment.copy(y, JAVA_FLOAT, 0, output, 0, count);
    } catch (Throwable error) { throw new IllegalStateException("Oracle attention failed", error); }
    state.positions()[0] = start+batch;
    calls++;
  }

  public void reset() { layers.values().forEach(state -> state.positions()[0] = 0); }
  public void rewind(int checkpoint) {
    if (checkpoint < 0) throw new IllegalArgumentException("Negative checkpoint");
    layers.values().forEach(state -> state.positions()[0] = Math.min(checkpoint, state.positions()[0]));
  }
  public void close() {
    System.err.println("Oracle attention calls: " + calls);
    layers.clear();
    arena.close();
  }
}
