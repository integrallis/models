#!/usr/bin/env python3
"""Train a LoRA that reads its options from the prompt, on Qwen3-0.6B.

Every frozen-probe variant measured on 2026-09-21 could separate a fixed class set it was trained
on and nothing more: Score collapsed past three ordinal levels under two prompts and two losses,
and a candidate-alignment head scored exactly chance on an option set it had not seen. The common
factor was reading a representation rather than reshaping one.

This trains the adapter to emit the chosen option, with the option list rendered into each prompt
so nothing can be answered by memorising a label position. The emitted token is only the training
signal -- the shipped decision comes from a typed head over the adapted hidden state, so the model
still cannot generate an answer at inference.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, random, time
from pathlib import Path

BASE = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def render(record: dict) -> str:
    options = "\n".join(f"- {o}" for o in record["options"])
    return (f"{record['state']}\n\nChoose exactly one option:\n{options}\nAnswer:")

def build_example(tokenizer, record: dict, max_length: int, eos: str):
    prompt_ids = tokenizer(render(record), add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(" " + record["labelText"] + eos, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + completion_ids
    if len(ids) > max_length:
        return None
    # Loss on the completion only: the prompt is given, not predicted.
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + completion_ids}

def evaluate(model, tokenizer, records, device, limit=None):
    """Score by comparing the model's loss on each option, not by decoding free text.

    Decoding would measure whether the adapter emits a well-formed label, which is a property of
    the training objective rather than of the decision. Ranking the options by their own likelihood
    asks the question the head will ask.
    """
    import torch
    model.eval()
    rows = records[:limit] if limit else records
    hits, total = 0, 0
    per_set = {}
    with torch.no_grad():
        for r in rows:
            prompt_ids = tokenizer(render(r), add_special_tokens=False)["input_ids"]
            best, best_score = None, None
            for index, option in enumerate(r["options"]):
                option_ids = tokenizer(" " + option, add_special_tokens=False)["input_ids"]
                ids = torch.tensor([prompt_ids + option_ids], device=device)
                labels = torch.tensor(
                    [[-100] * len(prompt_ids) + option_ids], device=device)
                loss = model(input_ids=ids, labels=labels).loss.item()
                # Length-normalised, so a long option is not penalised for being long.
                score = -loss
                if best_score is None or score > best_score:
                    best, best_score = index, score
            hits += best == r["label"]
            total += 1
            bucket = per_set.setdefault(r["optionSet"], [0, 0])
            bucket[0] += best == r["label"]
            bucket[1] += 1
    model.train()
    return {"accuracy": hits / total if total else 0.0, "n": total,
            "perSet": {k: {"accuracy": v[0] / v[1], "n": v[1]} for k, v in per_set.items()}}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-tokens", type=int, default=8000)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--eval-limit", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260921)
    a = p.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    eos = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, revision=BASE_REVISION, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    config = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                        target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    model = get_peft_model(model, config); model.print_trainable_parameters()

    train_records = [json.loads(l) for l in (a.prepared / "train.jsonl").open()]
    validation_records = [json.loads(l) for l in (a.prepared / "validation.jsonl").open()]
    holdout_records = [json.loads(l) for l in (a.prepared / "holdout.jsonl").open()]
    trained_sets = sorted({r["optionSet"] for r in train_records})
    holdout_sets = sorted({r["optionSet"] for r in holdout_records})
    assert not (set(trained_sets) & set(holdout_sets)), "holdout option set leaked into training"
    print(f"trained option sets {trained_sets}  held out {holdout_sets}", flush=True)

    examples, dropped = [], 0
    for r in train_records:
        e = build_example(tokenizer, r, a.max_length, eos)
        if e is None: dropped += 1
        else: examples.append(e)
    print(f"train examples {len(examples)} (dropped {dropped})", flush=True)

    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
    batches, cur, cur_tokens = [], [], 0
    for i in order:
        n = len(examples[i]["input_ids"])
        if cur and (max(cur_tokens, n) * (len(cur) + 1) > a.batch_tokens):
            batches.append(cur); cur, cur_tokens = [], 0
        cur.append(i); cur_tokens = max(cur_tokens, n)
    if cur: batches.append(cur)

    steps_per_epoch = math.ceil(len(batches) / a.gradient_accumulation)
    total_steps = math.ceil(steps_per_epoch * a.epochs)
    optimizer = torch.optim.AdamW(
        [q for q in model.parameters() if q.requires_grad], lr=a.learning_rate, weight_decay=0.0)
    warmup = max(1, total_steps // 20)
    def lr_at(step):
        if step < warmup: return a.learning_rate * step / warmup
        return a.learning_rate * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    a.out.mkdir(parents=True, exist_ok=True)
    logf = (a.out / "training-log.jsonl").open("w")
    print(f"batches/epoch {len(batches)} optimizer steps {total_steps} warmup {warmup}", flush=True)

    # Before any training, so the adapter's contribution is a difference rather than a claim.
    baseline = evaluate(model, tokenizer, holdout_records, device, a.eval_limit)
    print(f"  holdout BEFORE training: {baseline['accuracy']:.4f} (n={baseline['n']})", flush=True)
    logf.write(json.dumps({"event": "baseline", "holdout": baseline}) + "\n"); logf.flush()

    model.train(); step = 0; micro = 0; t0 = time.time()
    while step < total_steps:
        random.shuffle(batches)
        for batch in batches:
            width = max(len(examples[i]["input_ids"]) for i in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long)
            labels = torch.full((len(batch), width), -100, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, i in enumerate(batch):
                e = examples[i]; n = len(e["input_ids"])
                ids[row, :n] = torch.tensor(e["input_ids"])
                labels[row, :n] = torch.tensor(e["labels"])
                mask[row, :n] = 1
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                        labels=labels.to(device))
            (out.loss / a.gradient_accumulation).backward()
            micro += 1
            if micro % a.gradient_accumulation == 0:
                for group in optimizer.param_groups: group["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(
                    [q for q in model.parameters() if q.requires_grad], 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    print(f"  step {step}/{total_steps} loss {out.loss.item():.4f} "
                          f"{time.time() - t0:.0f}s", flush=True)
                    logf.write(json.dumps({"step": step, "loss": out.loss.item()}) + "\n"); logf.flush()
                if step >= total_steps: break

    validation = evaluate(model, tokenizer, validation_records, device, a.eval_limit)
    holdout = evaluate(model, tokenizer, holdout_records, device, a.eval_limit)
    print(f"\n  validation (trained sets) {validation['accuracy']:.4f}  {validation['perSet']}")
    print(f"  holdout BEFORE {baseline['accuracy']:.4f} -> AFTER {holdout['accuracy']:.4f}  (n={holdout['n']})")

    model.save_pretrained(str(a.out))
    manifest = {
        "schemaVersion": 1, "kind": "choice-option-set-lora",
        "base": {"model": BASE, "revision": BASE_REVISION},
        "adapter": {"rank": a.rank, "alpha": a.alpha, "dropout": a.dropout,
                    "targetModules": TARGET_MODULES},
        "data": {"trainedOptionSets": trained_sets, "heldOutOptionSets": holdout_sets,
                 "trainRows": len(examples), "dropped": dropped},
        "results": {"holdoutBefore": baseline, "validationAfter": validation,
                    "holdoutAfter": holdout},
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
    }
    (a.out / "training-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  adapter written to {a.out}")

if __name__ == "__main__":
    main()
