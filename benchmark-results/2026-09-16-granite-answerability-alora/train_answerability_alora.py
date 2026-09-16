#!/usr/bin/env python3
"""Train an activated LoRA (aLoRA) answerability specialist on Granite 4.1 3B.

The adapter has IBM's shape (rank 16, alpha 32, all seven projections, invocation
``<|start_of_role|>assistant<|end_of_role|>``) so the packaged bundle drops into the same runtime
path. Each record is rendered exactly as the qualification runner renders a window case (chat
template with ``documents`` and the generation prompt), the completion is the JSON string of the
label followed by end-of-text, and the loss covers only the completion tokens. The validation
split is scored by greedy decoding under the runner's strict contract.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, random, time
from pathlib import Path

BASE = "ibm-granite/granite-4.1-3b"
BASE_REVISION = "c0650403e44e78ec0262dab1c90914c65b196c4e"
INVOCATION_TOKENS = [100264, 78191, 100265]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LABELS = ("answerable", "unanswerable")

def render(tokenizer, record: dict) -> str:
    messages = [{"role": m["role"], "content": m["text"]} for m in record["messages"]]
    documents = [{"doc_id": d["doc_id"], "text": d["text"]} for d in record["documents"]]
    return tokenizer.apply_chat_template(messages, documents=documents, add_generation_prompt=True, tokenize=False)

def build_example(tokenizer, record: dict, max_length: int, eos: str) -> dict | None:
    prompt_ids = tokenizer(render(tokenizer, record), add_special_tokens=False)["input_ids"]
    if prompt_ids[-3:] != INVOCATION_TOKENS:
        raise ValueError(f"{record['id']}: prompt does not end with the invocation tokens")
    completion_ids = tokenizer(json.dumps(record["label"]) + eos, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + completion_ids
    if len(ids) > max_length:
        return None
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + completion_ids}

def strict_label(text: str) -> str | None:
    s = text.strip()
    for label in LABELS:
        if s == f'"{label}"': return label
    return None

def evaluate(model, tokenizer, records: list[dict], device, max_new_tokens: int = 6) -> dict:
    import torch
    # PEFT's activated-LoRA hooks refuse several forwards per backward while gradient
    # checkpointing is on, and generate() is many forwards: checkpointing off for the pass.
    base = model.base_model.model if hasattr(model, "base_model") else model
    base.gradient_checkpointing_disable(); base.config.use_cache = True
    model.eval(); correct = {l: 0 for l in LABELS}; total = {l: 0 for l in LABELS}; structured = 0; by_source = {}
    with torch.no_grad():
        for r in records:
            ids = tokenizer(render(tokenizer, r), add_special_tokens=False, return_tensors="pt").to(device)
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, pad_token_id=tokenizer.eos_token_id, logits_to_keep=1)
            gen = tokenizer.decode(out[0, ids["input_ids"].shape[1]:].tolist(), skip_special_tokens=False)
            gen = gen.split(tokenizer.eos_token)[0] if tokenizer.eos_token else gen
            pred = strict_label(gen); structured += pred is not None
            total[r["label"]] += 1; hit = pred == r["label"]; correct[r["label"]] += hit
            s = by_source.setdefault(r["source"], {"n": 0, "hit": 0}); s["n"] += 1; s["hit"] += hit
    base.gradient_checkpointing_enable(); base.config.use_cache = False
    model.train()
    bal = 0.5 * sum(correct[l] / max(total[l], 1) for l in LABELS)
    return {"balancedAccuracy": bal, "structuredRate": structured / max(len(records), 1),
            "answerable": [correct["answerable"], total["answerable"]], "unanswerable": [correct["unanswerable"], total["unanswerable"]],
            "bySource": {k: round(v["hit"] / v["n"], 4) for k, v in by_source.items()}}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=3072); p.add_argument("--rank", type=int, default=16); p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05); p.add_argument("--epochs", type=float, default=1.0); p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-tokens", type=int, default=12000, help="tokens per micro-batch (length-bucketed)")
    p.add_argument("--gradient-accumulation", type=int, default=4); p.add_argument("--train-limit", type=int); p.add_argument("--validation-limit", type=int)
    p.add_argument("--eval-every", type=int, default=200); p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA: nf4 base weights (bitsandbytes) for small GPUs; the adapter still applies to the full base")
    a = p.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    eos = tokenizer.eos_token
    if a.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, quantization_config=quant, device_map={"": 0}, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        # not prepare_model_for_kbit_training: it upcasts the embeddings and head to fp32
        # (~2 GB on this vocabulary), which an 8 GB card cannot spare; bf16 is fine for LoRA.
        for param in model.parameters():
            param.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    config = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
                        alora_invocation_tokens=INVOCATION_TOKENS)
    model = get_peft_model(model, config); model.print_trainable_parameters()
    train_records = [json.loads(l) for l in (a.prepared / "train.jsonl").open()][: a.train_limit or None]
    validation_records = [json.loads(l) for l in (a.prepared / "validation.jsonl").open()][: a.validation_limit or None]
    examples, dropped = [], 0
    for r in train_records:
        e = build_example(tokenizer, r, a.max_length, eos)
        if e is None: dropped += 1
        else: examples.append(e)
    print(f"train examples {len(examples)} (dropped {dropped} over {a.max_length} tokens) validation {len(validation_records)}", flush=True)
    # length-bucketed micro-batches: sort by length in chunks, pack up to batch-tokens
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
    batches, cur, cur_tokens = [], [], 0
    for i in order:
        n = len(examples[i]["input_ids"])
        if cur and (max(cur_tokens, n) * (len(cur) + 1) > a.batch_tokens):
            batches.append(cur); cur, cur_tokens = [], 0
        cur.append(i); cur_tokens = max(cur_tokens, n)
    if cur: batches.append(cur)
    steps_per_epoch = math.ceil(len(batches) / a.gradient_accumulation); total_steps = math.ceil(steps_per_epoch * a.epochs)
    optimizer = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=a.learning_rate, weight_decay=0.0)
    warmup = max(1, total_steps // 20)
    def lr_at(step): return a.learning_rate * (step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    log = (a.out / "training-log.jsonl"); a.out.mkdir(parents=True, exist_ok=True); logf = log.open("w")
    step, micro, t0, best = 0, 0, time.time(), None
    print(f"batches/epoch {len(batches)} optimizer steps {total_steps} warmup {warmup}", flush=True)
    model.train()
    epoch = 0
    while step < total_steps:
        random.shuffle(batches)
        for batch in batches:
            if step >= total_steps: break
            # Left-pad so every completion ends at the last position, then ask the model for the
            # logits of the last K positions only: the loss covers completion tokens alone, and the
            # full-vocabulary logits over the prompt (the dominant activation on a small GPU) are
            # never materialised. RoPE is relative, so the left shift does not change attention.
            L = max(len(examples[i]["input_ids"]) for i in batch)
            K = max(sum(1 for t in examples[i]["labels"] if t != -100) for i in batch) + 1
            ids = torch.full((len(batch), L), pad, dtype=torch.long); labels = torch.full((len(batch), L), -100, dtype=torch.long); mask = torch.zeros((len(batch), L), dtype=torch.long)
            for row, i in enumerate(batch):
                e = examples[i]; n = len(e["input_ids"]); ids[row, L - n:] = torch.tensor(e["input_ids"]); labels[row, L - n:] = torch.tensor(e["labels"]); mask[row, L - n:] = 1
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), logits_to_keep=K)
            logits = out.logits[:, :-1, :].float()                      # positions L-K .. L-2 predict tokens L-K+1 .. L-1
            targets = labels[:, L - K + 1:].to(device)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            (loss / a.gradient_accumulation).backward(); micro += 1
            loss_value = loss.item(); del out, logits, loss  # drop the graph before any validation pass
            if micro % a.gradient_accumulation == 0:
                for g in optimizer.param_groups: g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                rec = {"step": step, "loss": round(loss_value, 4), "lr": lr_at(step), "elapsed": round(time.time() - t0)}
                if step % 10 == 0: print(json.dumps(rec), flush=True)
                if step % a.eval_every == 0 or step == total_steps:
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    rec["validation"] = evaluate(model, tokenizer, validation_records, device); print(json.dumps(rec), flush=True)
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    if best is None or rec["validation"]["balancedAccuracy"] >= best["validation"]["balancedAccuracy"]:
                        best = rec; model.save_pretrained(str(a.out / "adapter"))
                logf.write(json.dumps(rec) + "\n"); logf.flush()
        epoch += 1
    manifest = {"base": BASE, "baseRevision": BASE_REVISION, "invocation": {"tokens": INVOCATION_TOKENS, "text": "<|start_of_role|>assistant<|end_of_role|>"},
                "config": {"rank": a.rank, "alpha": a.alpha, "dropout": a.dropout, "targetModules": TARGET_MODULES, "maxLength": a.max_length, "epochs": a.epochs,
                           "learningRate": a.learning_rate, "batchTokens": a.batch_tokens, "gradientAccumulation": a.gradient_accumulation, "seed": a.seed, "loadIn4bit": a.load_in_4bit},
                "prepared": json.loads((a.prepared / "manifest.json").read_text()), "trainExamples": len(examples), "droppedOverLength": dropped,
                "steps": total_steps, "best": best, "seconds": round(time.time() - t0),
                "versions": {"python": platform.python_version(), "torch": torch.__version__, "peft": __import__("peft").__version__, "transformers": __import__("transformers").__version__,
                             "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}}
    files = {f.name: {"bytes": f.stat().st_size, "sha256": sha256(f)} for f in sorted((a.out / "adapter").iterdir()) if f.is_file()}
    manifest["adapterFiles"] = files
    (a.out / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("DONE", json.dumps({"best": best, "files": list(files)}), flush=True)

if __name__ == "__main__":
    main()
