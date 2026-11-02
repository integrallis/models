set -e
export HOME=/root
. /root/.cargo/env
mkdir -p /root/bench && tar xzf /root/payload2.tgz -C /root/bench
mkdir -p /root/models-rag-bench && rm -rf /root/models-rag-bench/lib && mv /root/bench/lib /root/models-rag-bench/lib
chmod +x /root/bench/fleet/*.sh
cd /root/bench/crate
echo "### building FIXED kernel (rebased branch)"
cargo build --release >/dev/null 2>&1
cp target/release/libjmodels_kernels.so /root/libkernels-fixed.so
echo "### building MAIN kernel (baseline)"
cp src/lib.rs /root/lib.rs.fixed.keep
cp lib.rs.main src/lib.rs
cargo build --release >/dev/null 2>&1
cp target/release/libjmodels_kernels.so /root/libkernels-main.so
cp /root/lib.rs.fixed.keep src/lib.rs
ls -la /root/libkernels-*.so
for f in /root/libkernels-fixed.so /root/libkernels-main.so; do echo -n "$(basename $f) "; sha256sum $f | cut -c1-16; done
echo "### differ? $(cmp -s /root/libkernels-fixed.so /root/libkernels-main.so && echo IDENTICAL-PROBLEM || echo yes-distinct)"
echo "### BUILD DONE"
