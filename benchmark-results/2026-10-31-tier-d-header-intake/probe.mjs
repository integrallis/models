import { inspectGguf } from "/Users/briansam-bodden/Code/java-ai/projects/model-jars-central-gate/tools/gguf-inspector.mjs";
// #284 says: "tdt_ctc-110m-q4_k.gguf (0.13 GB) was sized but never probed. Its architecture is
// assumed from its filename, which is exactly the assumption the header intake exists to avoid.
// Probe its header before planning around it." Doing that, plus inflect-v2 for #286.
const TARGETS = [
  ["parakeet-110m",  "mudler/parakeet-cpp-gguf",         "tdt_ctc-110m-q4_k.gguf"],
  ["parakeet-0.6b",  "mudler/parakeet-cpp-gguf",         "tdt-0.6b-v3-q8_0.gguf"],
  ["inflect-v2",     "remixerdec/Inflect-Nano-v2-GGUF",  "inflect_nano_v2_q8_0.gguf"],
];
for (const [label, repo, file] of TARGETS) {
  const uri = `https://huggingface.co/${repo}/resolve/main/${encodeURI(file)}`;
  try {
    const head = await fetch(uri, { method: "HEAD", redirect: "follow" });
    const len = head.headers.get("x-linked-size") ?? head.headers.get("content-length");
    const { metadata, tensorInfos } = await inspectGguf(uri, {
      additionalFetchHeaders: { "User-Agent": "ModelJars-Catalog-Inspector/0.1" },
    });
    const roots = [...new Set(tensorInfos.map(t => t.name.split(".")[0]))].sort();
    console.log(`\n##### ${label}  HTTP ${head.status}  ${len} bytes  tensors ${tensorInfos.length}`);
    console.log(`  general.architecture = ${JSON.stringify(metadata["general.architecture"] ?? null)}`);
    console.log(`  general.name         = ${JSON.stringify(metadata["general.name"] ?? null)}`);
    console.log(`  TENSOR ROOTS: ${roots.join(" ")}`);
    const keys = Object.entries(metadata).filter(([k]) => !/^tokenizer\./.test(k));
    console.log(`  ${keys.length} non-tokenizer keys:`);
    for (const [k, v] of keys) {
      console.log(`    ${k} = ${Array.isArray(v) ? `[array ${v.length}]` : JSON.stringify(v)}`);
    }
  } catch (e) { console.log(`\n##### ${label}  FAILED: ${e.message}`); }
}
