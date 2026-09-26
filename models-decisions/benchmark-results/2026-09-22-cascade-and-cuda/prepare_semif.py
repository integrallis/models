"""Render JevBench rows exactly as jevbench/adapters/semif_direct.py does, so our arm differs
from SemIf's only in the runtime (our Q4_K_M Java engine vs their BF16 PyTorch).

Faithful to the adapter, which a first attempt was not:
  * every option description is prefixed "<id>: "
  * noul options are ("true", "false") IN THAT ORDER, description = criteria[k] or
    "The proposition is <k>."; the adapter maps true->yes and false->no on the way out
  * choice options follow criteria key order, not the labels array
  * score options are "0".."k-1" with the level text
  * evidence carries task.state as a nested JSON value, not a pre-stringified copy

The chat wrapper is our own qwen4b.gguf tokenizer.chat_template rendered with jinja2 under
add_generation_prompt=True, enable_thinking=False, reproduced verbatim.
"""
import json, sys

BENCH = "/Users/briansam-bodden/Code/java-ai/references/openjev/gh-jevbench/datasets/public"
SYS = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
       "Respond with only its uppercase letter, with no explanation or reasoning.")
LETTERS = "ABCDEFGHIJKLMNOP"

def build(task):
    q = task["question"]
    qtype, crit = q["type"], q.get("criteria")
    if qtype == "noul":
        options = [{"id": k, "description": (crit or {}).get(k, f"The proposition is {k}.")}
                   for k in ("true", "false")]
        out_labels = ["yes", "no"]               # adapter maps true->yes, false->no
    elif qtype == "choice":
        options = [{"id": k, "description": v or k} for k, v in crit.items()]
        out_labels = [o["id"] for o in options]
    else:
        options = [{"id": str(i), "description": lvl} for i, lvl in enumerate(crit)]
        out_labels = [o["id"] for o in options]
    for o in options:
        o["description"] = o["id"] + ": " + o["description"]
    return options, out_labels, q["instructions"]

out = open(sys.argv[1], "w")
n = 0
for split in ("easy", "original", "hard"):
    for line in open(f"{BENCH}/{split}.jsonl"):
        t = json.loads(line)
        options, out_labels, criterion = build(t)
        payload = {
            "evidence": t["state"],                      # nested value, as the adapter passes it
            "criterion": criterion,
            "options": [{"letter": LETTERS[i], "description": o["description"]}
                        for i, o in enumerate(options)],
        }
        prompt = (
            "<|im_start|>system\n" + SYS + "<|im_end|>\n"
            "<|im_start|>user\n" + json.dumps(payload, ensure_ascii=False) + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        out.write("\t".join([t["id"], t["question"]["type"], "|".join(out_labels),
                             prompt.replace("\\", "\\\\").replace("\t", " ").replace("\n", "\\n")]) + "\n")
        n += 1
out.close()
print(f"prepared {n} rows -> {sys.argv[1]}")
