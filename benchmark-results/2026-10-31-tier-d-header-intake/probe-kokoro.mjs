import { inspectGguf } from "/Users/briansam-bodden/Code/java-ai/projects/model-jars-central-gate/tools/gguf-inspector.mjs";
const uri = "https://huggingface.co/simonfxr/kokoro.cpp-GGUF/resolve/main/kokoro-82m-f16.gguf";
const { metadata } = await inspectGguf(uri, { additionalFetchHeaders: { "User-Agent": "ModelJars-Catalog-Inspector/0.1" } });
for (const [k,v] of Object.entries(metadata)) {
  const s = Array.isArray(v) ? `[array ${v.length}]` : JSON.stringify(v);
  console.log(`${k} = ${s}`);
}
