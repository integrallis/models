export HOME=/root
set -e
CP=$(ls /root/models-rag-bench/lib/*.jar | tr '\n' ':')
mkdir -p /root/kbench/classes
/root/jdk/bin/javac -nowarn -cp "$CP" -d /root/kbench/classes /root/kbench/Q4KThroughputProbe.java 2>&1 | head -5
echo "### one JVM per thread count -- no leftover worker pools"
for t in 2 4 8 12 16 8 16; do
  /root/jdk/bin/java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
    -Xmx6g -Dmodels.native.kernels.library=/root/model-kernels/target/release/libjmodels_kernels.so \
    -cp "$CP:/root/kbench/classes" Q4KThroughputProbe $t 2>&1 | grep -E '^threads='
done
