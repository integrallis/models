"""tasks.jsonl -> flat TSV the Java runner reads. Keeps JSON handling out of Java."""
import json, sys

src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(dst, "w") as out:
    for line in open(src):
        t = json.loads(line)
        q = t["question"]
        crit = q.get("criteria")
        labels = t["labels"]
        # The rubric every adapter sends, never less of it.
        if q["type"] == "noul":
            c = crit or {}
            rubric = {"no": c.get("false", "No"), "yes": c.get("true", "Yes")}
        elif q["type"] == "score":
            rubric = dict(zip(labels, crit))
        else:
            rubric = {k: (v or k) for k, v in crit.items()}
        # Jev accepts a structured state natively; a text model needs it rendered. Serialising as
        # compact JSON is the least lossy rendering available and is applied uniformly, but it is a
        # handicap on those items and is recorded as one rather than hidden.
        state = t["state"]
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        lines = [state_text, "", q["instructions"], "Options:"]
        for lab in labels:
            lines.append(f"- {lab}: {rubric.get(lab, lab)}")
        lines.append("Answer:")
        prompt = "\n".join(lines)
        out.write("\t".join([t["id"], q["type"], "|".join(labels),
                             prompt.replace("\t", " ").replace("\n", "\\n")]) + "\n")
        n += 1
print(f"prepared {n} tasks -> {dst}")
