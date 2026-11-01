# ARM SLEEF activation integration candidate

This external candidate replaces the ARM128 activation recipe with explicit SLEEF EXP/LOG operation order matching the pinned published warmed ARM JDK. The code is Boost-1.0 licensed. X86 arithmetic is unchanged. Native sigmoid/softplus and convolution use optional provider calls in all Qwen3.5 graph paths; unsupported width or library capability retains Java. Caller-owned array views do not extend the scratch or session lifetime.

The standalone numerical proof covered 878,137,347 comparisons, including all retained real inputs, with no mismatches. The v3 component comparison was faster than published in all four measured shapes with zero warm allocation, but standalone cold allocation failed and convolution precision remained unresolved. These results do not qualify this composed library. Actual-provider numerical, task quality, load, first-request, latency, allocation and RSS gates remain pending for the exact composition and every retained platform.

Raw prototypes and their failed controls are outside the repositories under `audit-artifacts/sleef-native-arm-v3` and `audit-artifacts/sleef-models-arm-v1`. No benchmark-only entry points or standalone library-loader lifecycle are included here.
