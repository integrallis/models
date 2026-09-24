# Reproducing the runs in NOTES.md

The host they ran on was torn down afterwards. Everything below rebuilds it. Roughly
an hour, most of it the model download.

## Host

Hetzner CCX33 (dedicated vCPU): 8 vCPU on AMD EPYC Milan, 4 physical cores with SMT,
32 GiB, Ubuntu 24.04, one region. Dedicated rather than shared vCPU matters: the
thread-scaling result in section 1 is only readable on cores nobody else is using.

Nothing else may run on the box during a timed run. The thread-scaling and grouping
numbers are wall-clock ratios and a second JVM invalidates all of them.

## Toolchain

    # JDK 25 (Temurin) into ~/jdk
    curl -sL <temurin-25-linux-x64-tarball> | tar xz && mv jdk-25* ~/jdk

    # Rust, to build the kernel for this platform. No cross-compile: the shipped
    # artifact carries a .so per platform, and the ABI here is 6.
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    rsync -a --exclude target models/backend-native/src/main/rust/model-kernels/ ~/kernels/
    cd ~/kernels && cargo build --release

## Layout

    ~/v53/
      lib/        every jar of the dependency closure, flat
      native/     META-INF/models/native/<platform>/{libjmodels_kernels.so,native.properties}
      classes/    compiled harnesses
      src/demo/   the harnesses in harness/
      document.txt cases.tsv jevbench-120.tsv

`native.properties` for `linux-x86_64` must declare the library actually present:

    abi=6
    platform=linux-x86_64
    library=libjmodels_kernels.so
    sha256=<sha256sum of the .so>

`BundledNativeKernelLibrary` verifies both the ABI and the digest, so a stale
properties file is a load failure and not a silent fallback to Java. Confirm the
library really loaded before trusting a number:

    java ... demo.Probe document.txt | grep native-kernel-abi     # must print 6

## Warmup, which is part of the measurement

Every harness in `harness/` warms each path it is about to compare, and none of the
numbers in NOTES.md can be reproduced without that. Unwarmed Panama vector code takes
a different path with a different accumulation order, so an unwarmed comparison of two
code paths measures which one the compiler reached first. The first pass of this work
reported a state leak that did not exist for exactly that reason; see NOTES.md section 0.

Check the premise before trusting a comparison:

    java ... demo.Determinism <model>        # every line must read 0.000e+00

and if a harness is modified so that it exercises a new path, warm that path too.

## Running

    CP="classes:native:lib/*"
    JAVA="$HOME/jdk/bin/java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector"

    # Section 1: is prefill compute bound?
    for t in 1 2 4 8; do
      $JAVA -Dmodels.native.gatedDeltaNet=true \
            -Dmodels.native.kernels.threads=$t -Dvectors.gguf.threads=$t \
            -cp "$CP" demo.Threads document.txt
    done

    # Section 1: prefill against token count, and the parts of one decision
    $JAVA -Dmodels.native.gatedDeltaNet=true -cp "$CP" demo.Split document.txt

    # Section 2: grouped against one at a time, both arms measured
    $JAVA -Dmodels.native.gatedDeltaNet=true -cp "$CP" demo.Batch2 document.txt
    $JAVA -Dmodels.native.gatedDeltaNet=true -Dmodels.native.quantizedDecode=false \
          -cp "$CP" demo.Batch2 document.txt

    # Section 3: batched against per-token arithmetic
    $JAVA -Dmodels.native.gatedDeltaNet=true -cp "$CP" demo.Chunk document.txt

    # Section 4: the options block, latency only
    $JAVA -Dmodels.native.gatedDeltaNet=true -cp "$CP" demo.Reorder cases.tsv

Note that `demo.Batch2` calls `decideAll`, whose grouping is gated on what the backend
reports for `groupedDecisionBreakEven()`. To force the grouped path where the backend
says it does not pay, add `-Dmodeljars.decisions.minimumGroupSize=2`.

    # Section 4 and the answer-changing comparisons
    $JAVA -cp "$CP" demo.Cross document.txt      # batched, stepped, split, both kernels
    $JAVA -cp "$CP" demo.Matmul <model> blk.0.ffn_up.weight
    $JAVA -cp "$CP" demo.Grouped <model>         # add -Ddiag.kernel=java for the other arm

## Prompt arms against gold labels

JevBench v1.2's public items, from its own repository:

    cat datasets/public/easy.jsonl datasets/public/original.jsonl > jevbench-120.jsonl
    python3 options-first/prepare_tasks.py jevbench-120.jsonl jevbench-120.tsv

Every arm, one at a time, temperature 1.0, nothing fitted. `JevBenchRunner` warms
three items and discards them; that is verified sufficient by re-running at 25 and
comparing every column but latency.

    M=~/.modeljars/cache/sha256/00/00fe7986...a4/model.gguf
    B="$JAVA -Xmx12g -Dmodels.purejava.maxContextLength=8192 \
        -Dmodels.native.quantizedDecode=true -Ddecisions.letterLogits=true \
        -cp $CP com.integrallis.models.decisions.JevBenchRunner $M jevbench-120.tsv"

    $B shipped-w3.tsv 1.0                                          # benchmark prompt
    $B shipped-w25.tsv 1.0 -Ddecisions.warmupItems=25               # warmup control
    $B shipped-gdnnative.tsv 1.0 -Dmodels.native.gatedDeltaNet=true # noise floor
    $B runtime-prompt.tsv 1.0 -Ddecisions.runtimePrompt=true        # no rubric
    $B runtime-block.tsv 1.0 -Ddecisions.runtimeCriteria=true -Ddecisions.rubricStyle=block
    $B runtime-criteria.tsv 1.0 -Ddecisions.runtimeCriteria=true    # inline rubric
    $B options-first-fast.tsv 1.0 -Ddecisions.optionsFirst=true

Note that the properties must precede the class name; the line above is folded for
reading. `runtime-renderer.tsv` is `-Ddecisions.runtimeCriteria=true` taken after the
block layout became the renderer's default, and must equal `runtime-block.tsv`.

Scored with the benchmark's own modules, not a reimplementation:

    python3 options-first/score_arm.py jevbench-120.jsonl \
      prompt-arms/shipped-w3.tsv shipped \
      prompt-arms/options-first-fast.tsv options-first

`score_arm.py` expects a checkout of JevBench on its `sys.path`; the path is at the top
of the file. Digests of both arms' outputs are in `options-first/SHA256SUMS`.

## Model

`unsloth.qwen3.5-4b-gguf.q4_k_m` at marker version `3.5.0-q4_k_m.2`, resolved through
ModelJars, which verifies the digest on open. 2.55 GiB. The composite recipe is
`org.modeljars.composite:harriet`.
