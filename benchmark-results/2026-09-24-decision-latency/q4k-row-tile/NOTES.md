# Where a forward pass actually loses its time

Measured 2026-09-24 on a dedicated Hetzner CCX33 (AMD EPYC Milan / Zen 3, 8 vCPU on 4
physical cores, 30 GiB, Ubuntu 24.04), one arm at a time, nothing else on the box.
Model `unsloth/Qwen3.5-4B-GGUF` Q4_K_M, sha256 `00fe7986...a4`, the same artifact the
shipped harriet marker pins. Kernel built on the host; ABI 6; digest checked by
`BundledNativeKernelLibrary` so a stale library is a load failure, not a silent
fallback to Java.

Everything below is *measured here*. Nothing is read from a paper, a vendor page, or
carried over from the host the earlier runs used.

## 0. Two things I had wrong going in

**The clock.** `lscpu` reports `NotSpecified CPU @ 2.0GHz`. The cores actually run at
**2400 MHz**, verified under an eight-way busy load. Every per-cycle figure in earlier
notes that used 2.0 GHz is 20% too generous.

**"The inner loop is at its algorithmic ceiling."** Written in an earlier pass after
two failed experiments. It is wrong. The same kernel reaches **25.6 MAC/cycle/core**
when its weights are cache resident and **17.1** when they stream. The arithmetic is
not the wall.

A third, smaller one: the symptom that motivated the row-tiling work below --
`ffn_up` at 445-478 G-MAC/s against `ffn_down` at 244, a 1.9x gap at identical MAC
counts -- **does not reproduce on this host**. Here the same two shapes read 204 and
174, a 1.2x gap, at less than half the absolute rate. That measurement came from a
host that no longer exists and should not have been treated as a standing fact.

## 1. The headline: it is not bandwidth, and it is not arithmetic

| | G-MAC/s | MAC/cycle/core |
|---|---|---|
| matmul, weights cache resident | 246 | 25.6 |
| matmul, weights cold | 164 | 17.1 |
| whole forward pass, 18 tokens | 150 | 15.6 |

Achievable memory read bandwidth on this box (`bw.c`, 3 GiB buffer, best of 3):

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| GB/s | 33.9 | 37.3 | 38.0 | 37.9 |

Q4_K spends 144 bytes per 256 weights, and at batch 18 each weight byte feeds 18
multiply-accumulates: **32 MAC per weight byte**. So 164 G-MAC/s needs 5.1 GB/s and
246 G-MAC/s needs 7.7 GB/s. Against 38 GB/s available, both arms use **under a fifth
of the memory system**.

So the forward pass is not bandwidth bound, and it is not at the arithmetic ceiling
either, because the identical code goes 50% faster when the weights happen to be in
L3. What is left is **load latency** -- the cores stall waiting on weight lines that
the prefetcher and the TLB did not have ready.

That is the finding. It says the remaining performance is in how weights reach the
core, not in how many operations the inner loop does, and it invalidates the class of
optimisation that the previous two attempts belonged to.

## 2. Q4_K output-row tiling: +9.1% isolated, +0.4% real. Rejected.

The idea: walk output rows in tiles so each block's activations are read by a whole
tile while still in L1, instead of once per row. Bit-exact by construction -- each row
still accumulates its blocks in ascending order -- and `cargo test --release` agreed:
26 passed, 0 failed.

Tile size sweep, whole-model matmul floor, six rounds per arm, alternating:

| tile | 4 | 8 | 16 | 32 | **64** | 96 | 128 | 192 |
|---|---|---|---|---|---|---|---|---|
| vs stock | -0.1% | +7.6% | +7.3% | +3.9% | **+9.1%** | +8.4% | +6.1% | +8.5% |

Tile 64 reproduced across two independent sweeps at +9.2% and +9.1%. Taking the
winner to progressively more realistic measurements:

| measurement | stock | tile 64 | gain |
|---|---|---|---|
| one exemplar tensor per shape, 8 warmups + 10 rounds deep | 0.3549 s | 0.3227 s | +9.1% |
| every `blk.*` weight once, in layer order, cold (`ColdShape`) | 0.3920 s | 0.3795 s | +3.2% |
| real forward pass, 18 tokens (`Achieved`) | 0.4980 s | 0.4961 s | **+0.4%** |

**Reverted.** +0.4% over eight alternating rounds is noise: the within-arm spread is
0.468-0.522 s, an order of magnitude wider than the difference between the arms.

The mechanism is worth keeping even though the change is not. The stock loop is
`for row { for block }`, so weight reads march straight through the tensor. Tiling
makes it `for block { for row }`, so for a fixed block consecutive reads are a
**1440-byte stride** apart (`blocks_per_row` 10 x `Q4_K_BLOCK_BYTES` 144). When the
tensor is already in L3 the stride costs nothing and the activation reuse is pure
profit; when it streams from DRAM the stride defeats the prefetcher and gives the
gain straight back.

**This is why `PerShape` is a trap.** It times one exemplar tensor per shape, eight
warmups and ten rounds deep. A 9216x2560 Q4_K tensor is ~13 MB and this CCX has 32 MB
of L3, so by the time it is timed it is resident. A forward pass never sees a weight
twice -- it streams all 2.6 GB once. `ColdShape` was written for exactly this and any
future kernel change should be judged on it and on `Achieved`, never on `PerShape`
alone.

## 3. Huge pages: +3.7% on a forward pass, nothing on a decision

2.6 GB of weights under 4 KiB pages is 650k page-table entries against an L2 TLB that
holds roughly two thousand, so nearly every new block pays a page walk. Staging the
model on tmpfs with `shmem_enabled=force` backs it with 2 MiB pages instead;
`ShmemHugePages: 2693120 kB` in `/proc/meminfo` confirms the whole model was covered.

| measurement | 4 KiB pages | 2 MiB pages | gain |
|---|---|---|---|
| cold matmul (`ColdShape`), 4 rounds | 0.4033 s (159.1 G-MAC/s) | 0.3796 s (169.3) | +5.9% |
| forward pass (`Achieved`), 6 rounds | 0.4935 s | 0.4700 s | **+4.8%** |

Five of six rounds won. Bit-exact by construction: not one instruction changed, only
the page size behind the same bytes.

This recovers about a sixth of the resident-vs-cold gap from section 1, which both
confirms the latency diagnosis and says most of that gap is something other than the
TLB.

### 3a. Implemented, and it does not carry to a decision

`GgufHugePages` reads the weights into an anonymous `MADV_HUGEPAGE` mapping instead of
mapping the GGUF, so the win no longer depends on the file being on tmpfs. Measured on
the real implementation:

| measurement | 4 KiB pages | 2 MiB pages | |
|---|---|---|---|
| forward pass, 18 tokens, 8 rounds | 0.4999 s | 0.4813 s | **+3.7%**, 7/8 rounds |
| **video-shape decision, 10 rounds** | **1.3837 s** | **1.3689 s** | **+1.1%, 5/10 rounds** |

Per-round deltas on the decision arm: +0.062, +0.082, -0.044, +0.098, -0.112, -0.099,
+0.111, +0.001, -0.162, -0.085. That is a coin flip.

**This is no effect, not no data.** `AnonHugePages` was sampled at 2,674,688 kB during
the decision runs, so the treatment was applied. The default is therefore `off`, the
property stays, and both numbers are in the class javadoc.

The untested guess at why the split exists is arithmetic intensity. A decision prefills
a whole document and so runs at a much wider batch than 18 tokens; each weight byte then
feeds many more multiply-accumulates, and load latency -- the thing huge pages hide -- is
a smaller share of a compute-bound prefill. Recorded as a hypothesis, not a finding.

### 3b. Getting it wrong twice, both caught by checking the premise

**`Arena.allocate` zero-initialises.** The first implementation took the destination from
the arena and then advised it. Arena allocation returns zeroed memory, so every page was
already faulted in as 4 KiB before `madvise` ran, and advising afterwards only leaves the
region for khugepaged to maybe collapse later. Result: 16 MiB of huge pages out of 2.6 GiB
and a forward-pass A/B that measured -0.6%. The fix is to `mmap` the region directly, advise
it while it is still untouched, and let the arena own it via `reinterpret`. That also deleted
a 2.7 GB memset: parse went from 1.858 s to 0.653 s.

**`invokeExact` in statement position.** `madvise.invokeExact(...)` as a bare statement is a
`void` call against an `int` descriptor. It throws `WrongMethodTypeException` on every
invocation, which the surrounding catch-all swallowed, so the advice never happened and the
A/B would have compared 4 KiB pages against 4 KiB pages. Bind the result.

Both were caught for the same reason: the harness sampled `/proc/meminfo` and asserted the
pages were actually granted. A toggle that silently does nothing measures as "this does not
matter", which is the most expensive kind of wrong answer available here.

## 4. What this rules in and out for the next attempt

**Ruled out, with numbers:** anything that trades memory order for arithmetic economy.
Row tiling is the second such attempt to measure well in isolation and vanish on the
real workload. The inner loop is not the constraint.

**Ruled out, infrastructurally:** VNNI (`vpdpbusd`) needs Zen 4+ or Ice Lake+ and
Hetzner offers only Zen 3 in the dedicated CCX line.

**Ruled in, measured, shipped off by default:** huge-page-backed weights. Real on a bare
forward pass, absent on a decision.

**Ruled in, untested:** software prefetch at a tuned distance in the Q4_K row loop,
which attacks the measured stall directly. And -- more promising because we are using
17% of the memory system -- trading bytes for compute: pre-decoding Q4_K into a
cheaper in-memory layout such as int8 with a folded per-group scale. That roughly
doubles bytes read, which we demonstrably have room for, and deletes the nibble
unpack and the second scale multiply from the inner loop.

## Reproducing

Host, toolchain and layout as in `../REPRODUCE.md`. Additionally:

    # achievable bandwidth
    gcc -O2 -pthread -o /tmp/bw bw.c && /tmp/bw

    # cold matmul, the honest arm
    $JAVA -cp "classes:native-stock:lib/*" demo.ColdShape ~/model.gguf

    # forward pass. -Xmx and a bounded context are both required: the default heap
    # OOMs in SessionState, which allocates KV for the model's full declared context.
    $JAVA -Xmx20g -Dmodels.purejava.maxContextLength=4096 \
          -Dmodels.native.gatedDeltaNet=true \
          -cp "classes:native-stock:lib/*" demo.Achieved ~/model.gguf

    # huge pages
    echo force > /sys/kernel/mm/transparent_hugepage/shmem_enabled
    cp ~/model.gguf /dev/shm/model.gguf   # then run against /dev/shm/model.gguf

Raw logs for every arm above are beside this file: `per-shape.log`, `sweep2.log`,
`sweep3.log`, `fwd.log`, `cold.log`, `thp.log`, `thpf.log`.

### A trap that cost time here

`pgrep -f ab2.sh` in a wait loop matches **the wait loop's own command line**, so the
loop never exits and the run looks like it is still going hours after it finished. The
same shape of mistake killed an ssh session earlier in this project (`pkill -f
"while pgrep"`). Match on something the watcher itself cannot contain, or have the
script write a sentinel -- the scripts here echo `ALLDONE`.

Separately: six arms of the first forward-pass A/B produced no forward-pass numbers at
all, because stderr went to `/dev/null` and every one of them was dying on an
`OutOfMemoryError` in `SessionState`. Never discard stderr in a measurement harness.
