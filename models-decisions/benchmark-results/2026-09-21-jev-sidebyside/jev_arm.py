"""The Jev arm: same document, same questions, one call each, timed identically."""
import json, os, sys, time, urllib.request

doc = open(sys.argv[1]).read().strip()
questions = [q.strip() for q in open(sys.argv[2]) if q.strip()]
out_path = sys.argv[3] if len(sys.argv) > 3 else "jev-timing.json"
key = os.environ["TYPESAFE_API_KEY"]

print()
print("  TypeSafe Jev — one document, many questions")
print(f"  model jev-latest, {len(questions)} questions, document resent each call")
print()

lat, answers, in_tok = [], [], 0
start_all = time.perf_counter()
for i, q in enumerate(questions, 1):
    body = json.dumps({
        "state": doc,
        "model": "jev-latest",
        "questions": {"decision": {
            "type": "noul",
            "instructions": f"Is the question answerable from the text above? Question: {q}",
        }},
    }).encode()
    req = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        parsed = json.load(r)
    dt = time.perf_counter() - t
    p = parsed["answers"]["decision"]["noul"]
    in_tok += parsed.get("usage", {}).get("input_tokens", 0)
    lat.append(dt); answers.append(p)
    label = "YES" if p >= 0.5 else "NO "
    short = q if len(q) <= 46 else q[:43] + "..."
    print(f"  Q{i:<2} {short:<46} {label}  p={p:.3f}  {dt:6.3f} s")

total = time.perf_counter() - start_all
print()
print(f"  input tokens billed          {in_tok}   (document resent {len(questions)}x)")
print(f"  mean per question            {sum(lat)/len(lat):7.3f} s")
print(f"  total for {len(questions):<2} questions       {total:7.3f} s")
print()
json.dump({"system": "typesafe-jev", "model": parsed.get("model"),
           "questions": len(questions), "latency_s": lat, "probabilities": answers,
           "input_tokens_billed": in_tok, "total_s": round(total, 4)},
          open(out_path, "w"))
print(f"  timings written to {out_path}")
print()
