import { inspectGguf } from "/Users/briansam-bodden/Code/java-ai/projects/model-jars-central-gate/tools/gguf-inspector.mjs";

// WHY THIS EXISTS, and why the 2026-10-22 intake's absence claims cannot be trusted.
//
// inspectGguf skips every array-typed metadata key that is not named in `retainMetadataArrays`
// (gguf-inspector.mjs:88-93), and that option defaults to EMPTY. So a probe that does not pass it
// surfaces no arrays at all, and a key missing from its output means "absent OR array-typed" --
// two very different things. The 2026-10-22 intake did not pass it.
//
// This probe does two things that one did not:
//   1. retains every array key these architectures could plausibly use, and
//   2. ASSERTS surfaced == kv_count, printing the shortfall when it cannot account for every key.
// An absence is only reportable when the count reconciles.
const RETAIN = [
  // per-layer and per-architecture arrays
  "attention.head_count_kv", "attention.head_count", "attention.sliding_window_pattern",
  "expert_feed_forward_length", "swiglu_clamp_exp", "feed_forward_length",
  // the standard tokenizer arrays
  "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
  "tokenizer.ggml.scores", "general.tags", "general.languages", "general.datasets",
  "general.base_model.0.repo_url",
].flatMap((suffix) =>
  suffix.includes(".") && !suffix.startsWith("tokenizer") && !suffix.startsWith("general")
    ? ["mellum", "lfm2moe", "lfm2", "bailingmoe3", "kokoro", "inflect", "inflect-v2", "parakeet", "maple"]
        .map((arch) => `${arch}.${suffix}`)
    : [suffix]);

const TARGETS = [
  ["mellum-q4km",  "JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M", "Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf"],
  ["lfm2moe-q4_0", "LiquidAI/LFM2.5-8B-A1B-GGUF",                      "LFM2.5-8B-A1B-Q4_0.gguf"],
  ["kokoro",       "simonfxr/kokoro.cpp-GGUF",                         "kokoro-82m-f16.gguf"],
  ["inflect-v2",   "remixerdec/Inflect-Nano-v2-GGUF",                  "inflect_nano_v2_q8_0.gguf"],
  ["bailingmoe3",  "inclusionAI/Ling-3.0-tiny-GGUF",                   "Ling-3.0-tiny-Q4_K_M.gguf"],
];

for (const [label, repo, file] of TARGETS) {
  const uri = `https://huggingface.co/${repo}/resolve/main/${encodeURI(file)}`;
  const { metadata } = await inspectGguf(uri, {
    additionalFetchHeaders: { "User-Agent": "ModelJars-Catalog-Inspector/0.1" },
    retainMetadataArrays: RETAIN,
  });
  const synthetic = ["version", "tensor_count", "kv_count"];
  const real = Object.keys(metadata).filter((k) => !synthetic.includes(k));
  const gap = metadata.kv_count - real.length;
  console.log(`\n##### ${label}  kv_count=${metadata.kv_count}  surfaced=${real.length}  ` +
    (gap === 0 ? "RECONCILED — absences here are reportable"
               : `UNRECONCILED, ${gap} key(s) still skipped — NO absence claim is valid for this file`));
  for (const k of real.sort()) {
    const v = metadata[k];
    const shown = Array.isArray(v)
      ? `[array ${v.length}] ${JSON.stringify(v.slice(0, 10))}`
      : JSON.stringify(v);
    console.log(`  ${k} = ${String(shown).slice(0, 150)}`);
  }
}
