#!/usr/bin/env python3
"""Score a held-out option set at full size, with and without an adapter.

n=200 gives a binomial standard error of about 0.035, which cannot resolve the 0.03-0.045 deltas
the two training runs produced. The full 900 rows give about 0.017, which can.

Each arm loads the base weights **fresh**. An earlier version reused one base object and called
`unload()` between adapters; peft warned that a `peft_config` was already present, and a comparison
that might be scoring two stacked adapters is worth less than the time saved.
"""
import argparse, json, math, sys
from pathlib import Path

BASE = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

def render(r):
    options = "\n".join(f"- {o}" for o in r["options"])
    return f"{r['state']}\n\nChoose exactly one option:\n{options}\nAnswer:"

def score(model, tokenizer, records, device):
    import torch
    model.eval()
    flags = []
    with torch.no_grad():
        for r in records:
            prompt_ids = tokenizer(render(r), add_special_tokens=False)["input_ids"]
            best, best_score = None, None
            for index, option in enumerate(r["options"]):
                option_ids = tokenizer(" " + option, add_special_tokens=False)["input_ids"]
                ids = torch.tensor([prompt_ids + option_ids], device=device)
                labels = torch.tensor([[-100] * len(prompt_ids) + option_ids], device=device)
                s = -model(input_ids=ids, labels=labels).loss.item()
                if best_score is None or s > best_score:
                    best, best_score = index, s
            flags.append(best == r["label"])
    return sum(flags) / len(flags), flags

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--holdout", type=Path, required=True)
    p.add_argument("--adapters", nargs="+", required=True)
    a = p.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    records = [json.loads(l) for l in a.holdout.open()]
    counts = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    n = len(records)
    floor = max(counts.values()) / n
    print(f"  holdout {n} rows, {len(records[0]['options'])} options, floor {floor:.4f}\n", flush=True)

    def fresh_base():
        return AutoModelForCausalLM.from_pretrained(
            BASE, revision=BASE_REVISION, torch_dtype=torch.bfloat16).to(device)

    base = fresh_base()
    base_acc, base_flags = score(base, tokenizer, records, device)
    print(f"  base, no adapter         {base_acc:.4f}  "
          f"(SE {math.sqrt(base_acc * (1 - base_acc) / n):.4f})", flush=True)
    del base
    torch.cuda.empty_cache()

    for path in a.adapters:
        model = PeftModel.from_pretrained(fresh_base(), path)
        acc, flags = score(model, tokenizer, records, device)
        gained = sum(1 for b, f in zip(base_flags, flags) if f and not b)
        lost = sum(1 for b, f in zip(base_flags, flags) if b and not f)
        disc = gained + lost
        delta = acc - base_acc
        # McNemar on the discordant pairs: the right test for paired binary outcomes.
        pair_se = math.sqrt(disc) / n if disc else 0.0
        verdict = "SIGNIFICANT" if disc and abs(delta) > 2 * pair_se else "within noise"
        print(f"  {Path(path).name:24} {acc:.4f}  delta {delta:+.4f}  "
              f"gained {gained} lost {lost} discordant {disc}  "
              f"paired SE {pair_se:.4f}  -> {verdict}", flush=True)
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
