# Where the activation determinism defect actually lives

Every line reference below was read out of the working tree, not recalled. Verify with the stated
path and line before relying on any of it.

## The defect

`lanewise(VectorOperators.EXP)` changes its rounding once HotSpot intrinsifies the call, so a
model's first forward pass produces different floats from every later one. The cause is a property
of the JDK vector intrinsic, **not** of any model or any quantization. Any route reaching that call
inherits it.

One site additionally uses `lanewise(VectorOperators.LOG)`, a second intrinsic with the same
exposure, which the reproducible-exp helper does not address at all.

## The guard, and what it leaves exposed

`vectors-core/src/main/java/com/integrallis/vectors/core/PanamaVectorUtilSupport.java:6949`

```java
private static final boolean USE_PUBLISHED_ARM_ACTIVATIONS =
    "aarch64".equals(System.getProperty("os.arch")) && FLOAT_SPECIES.vectorBitSize() == 128;
```

One static boolean on architecture **and vector width** guards all three activation sites. When it
is true the call goes to `PublishedArmActivations` (the Arm optimized-routines polynomial) and is
deterministic. When it is false the vector-EXP route is taken and is not.

So determinism today is not "ARM is fine, x86 is not". It is:

| platform | route | deterministic |
| --- | --- | --- |
| aarch64, 128-bit (NEON) | `PublishedArmActivations` | yes |
| aarch64, wider than 128-bit (SVE, e.g. Graviton3) | `lanewise(EXP)` | **no** |
| x86-64, any width (AVX2, AVX-512) | `lanewise(EXP)` | **no** |

The width term splits aarch64 against itself. That is the part most easily missed, because "ARM is
deterministic" is true only at one vector width.

## Three sites, not one

| site | declared at | intrinsics on the non-ARM path |
| --- | --- | --- |
| `swiGlu` | `PanamaVectorUtilSupport.java:6953` | EXP at `:6973` |
| `sigmoidAndScaledSoftplus` | `:6984` | EXP at `:7019`, EXP+**LOG** at `:7025` |
| `causalDepthwiseConv1dSilu` | `:7047` | EXP at `:7098` |

## Which models reach which site

Callers, from `projects/models/backend-java/src/main/java` (grep for `.<name>(`):

- `swiGlu` — eight forward passes: `BertForwardPass`, `Deepseek2ForwardPass`, `Lfm2ForwardPass`,
  `Lfm2SequenceEncoder`, `EncoderForwardPass`, `LlamaForwardPass`, `QuantizedBatchedLayerPlan`,
  `Qwen35ForwardPass`. Effectively every architecture's FFN.
- `sigmoidAndScaledSoftplus` — `Qwen35ForwardPass` only.
- `causalDepthwiseConv1dSilu` — `Qwen35ForwardPass` only.

The architectures that route to `Qwen35ForwardPass` are declared, not inferred:
`Qwen35Config.java:65` has `ARCHITECTURES = List.of("qwen35", "qwen35moe", "qwen3next")`, and the
dispatch at `PureJavaBackend.java:498-500` matches those same three ids.

Catalog exposure, counted by the `architecture` field in `projects/model-jars/catalog/models.json`:
**462 entries total**, of which `qwen35` 31 + `qwen35moe` 4 + `qwen3next` 1 = **36** reach all three
sites. The other 426 reach only `swiGlu`. (Counting by a substring match over the whole record
instead gives 37, because it also catches `mobilemoe` and `gemma3n` on the word "gated"; the
architecture field is the right key.)

## What the candidate under test actually changes

`b3c1245d` ("EXPERIMENT: isolate the reproducible activation exp on v0.1.30") keeps all four
`USE_PUBLISHED_ARM_ACTIVATIONS` references that v0.1.30 has — verified by counting them in both
blobs — and replaces only the *else* branch of `swiGlu`. It is therefore already scoped to the
complement of the published-ARM route, and it covers **one of the three sites**.

Consequence: the candidate closes the defect for the 426 catalog entries that only
reach `swiGlu`, and leaves two of three sites open for the 36 Qwen3.5-family entries. A fix that
claims to make the runtime deterministic has to say which of those two scopes it means.

`ReproducibleActivationExp` reproduces `exp` only. Site 2 also needs a reproducible `log` before it
can be closed, which is additional unwritten work and not a line-move.

## What is measured and what is not

- Measured: throughput cost of the `swiGlu` substitution on x86-64 AVX2 (256-bit) with
  HuatuoGPT-o1-7B-Q4_K_M — see `articles/models-deep-dive/evidence/activation-cost-*`.
- Not measured: the same substitution on aarch64 above 128-bit, and either of the other two sites
  anywhere. Applying the change to those routes would be shipping an unmeasured change, which is a
  decision to take deliberately rather than by inheriting one static boolean.
