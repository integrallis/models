#!/usr/bin/env python3
"""Run every case of one window suite through the PEFT reference of an activated LoRA (CPU) and
score the answerability label from the greedy continuation. Measurement oracle only."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "/opt/ref")
from reference_alora_case import render  # same prompt rendering as the single-case oracle

def label_of(text: str) -> str | None:
    """The Java runner's specialist contract: the stripped completion is exactly the JSON string
    of one label (io.yaml). Anything else is unstructured."""
    stripped = text.strip()
    for label in ("answerable", "unanswerable"):
        if stripped == f'"{label}"':
            return label
    return None


def lenient_label_of(text: str) -> str | None:
    """What the completion meant, ignoring quoting: recorded beside the strict reading."""
    head = text.strip().strip('"').strip().lower()
    if head.startswith("unanswerable"): return "unanswerable"
    if head.startswith("answerable"): return "answerable"
    return None

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True); p.add_argument("--base-revision", required=True)
    p.add_argument("--adapter", type=Path, required=True); p.add_argument("--window", type=Path, required=True)
    p.add_argument("--suite", required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=8); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.base_revision)
    base = AutoModelForCausalLM.from_pretrained(a.base, revision=a.base_revision, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(a.adapter)); model.eval()
    window = json.loads(a.window.read_text())
    suite = next(s for s in window["suites"] if s["name"] == a.suite)
    cases = suite["cases"][: a.limit] if a.limit else suite["cases"]
    results = []; t0 = time.time()
    counts = {"answerable": [0, 0], "unanswerable": [0, 0]}; lenient_counts = {"answerable": 0, "unanswerable": 0}; structured = 0
    for i, case in enumerate(cases):
        text = render(tok, case)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:].tolist(), skip_special_tokens=True)
        pred = label_of(gen); structured += pred is not None; lenient = lenient_label_of(gen)
        counts[case["label"]][1] += 1; counts[case["label"]][0] += (pred == case["label"])
        lenient_counts[case["label"]] += (lenient == case["label"])
        results.append({"id": case["id"], "label": case["label"], "prediction": pred, "lenientPrediction": lenient, "generated": gen, "promptTokens": int(ids["input_ids"].shape[1])})
        print(f"{i+1}/{len(cases)} id={case['id']} label={case['label']} pred={pred} tokens={ids['input_ids'].shape[1]} gen={gen!r} elapsed={time.time()-t0:.0f}s", flush=True)
    ans = counts["answerable"]; un = counts["unanswerable"]
    bal = 0.5 * (ans[0] / max(ans[1], 1) + un[0] / max(un[1], 1))
    summary = {"suite": a.suite, "windowSha256": window["windowSha256"], "cases": len(cases), "structuredRate": structured / len(cases),
               "answerableCorrect": ans[0], "answerableCases": ans[1], "unanswerableCorrect": un[0], "unanswerableCases": un[1],
               "balancedAccuracy": bal, "lenientBalancedAccuracy": 0.5 * (lenient_counts["answerable"] / max(ans[1], 1) + lenient_counts["unanswerable"] / max(un[1], 1)),
               "dtype": "bfloat16", "seconds": round(time.time() - t0, 1)}
    a.output.write_text(json.dumps({"summary": summary, "cases": results}, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)

if __name__ == "__main__":
    main()
