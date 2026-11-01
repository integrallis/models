import { inspectGguf } from "/Users/briansam-bodden/Code/java-ai/projects/model-jars-central-gate/tools/gguf-inspector.mjs";
// The artifacts #281 and #283 actually name. The 2026-10-22 intake probed DIFFERENT quants of
// each (mellum Q8_0 from the -Instruct- repo, lfm2moe Q8_0), so the configs the two port issues
// plan against have never been read. "Sized but never probed" is the exact trap that intake's
// own README flags for parakeet's 110M sibling.
const TARGETS = [
  ["mellum-q4km-thinking", "JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M", "Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf"],
  ["lfm2moe-q4_0",         "LiquidAI/LFM2.5-8B-A1B-GGUF",                      "LFM2.5-8B-A1B-Q4_0.gguf"],
];
for (const [label, repo, file] of TARGETS) {
  const uri = `https://huggingface.co/${repo}/resolve/main/${encodeURI(file)}`;
  try {
    const head = await fetch(uri, { method: "HEAD", redirect: "follow" });
    const len = head.headers.get("x-linked-size") ?? head.headers.get("content-length");
    const { metadata, tensorInfos } = await inspectGguf(uri, {
      additionalFetchHeaders: { "User-Agent": "ModelJars-Catalog-Inspector/0.1" },
    });
    const roots = [...new Set(tensorInfos.map(t => t.name.replace(/^blk\.\d+\./, "blk.")))].sort();
    console.log(`\n##### ${label}  HTTP ${head.status}  ${len} bytes  tensors ${tensorInfos.length}`);
    for (const [k, v] of Object.entries(metadata)) {
      if (/^tokenizer\./.test(k)) continue;
      console.log(`  ${k} = ${Array.isArray(v) ? `[array ${v.length}]` : JSON.stringify(v)}`);
    }
    console.log(`  TENSOR ROOTS: ${roots.filter(r=>r.startsWith("blk.")).join(" ")}`);
    console.log(`  TOP: ${roots.filter(r=>!r.startsWith("blk.")).join(" ")}`);
  } catch (e) { console.log(`\n##### ${label}  FAILED: ${e.message}`); }
}
