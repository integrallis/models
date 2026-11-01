#!/usr/bin/env python3
"""Serve a Hugging Face transformers model over the `/completion` shape our harness speaks.

WHY THIS EXISTS. `RagBenchmarkCli --backend transformers` already exists and `transformers` is one
of `RagProductionQualificationPolicy.AUTHORITATIVE_ENGINES`, so a transformers arm can qualify a
model. What was missing was only the HTTP shape: `HttpGenerationClient` posts to `/completion` with
llama.cpp's field names and parses llama.cpp's streaming events, and `transformers` has no such
server. This is that adapter and nothing more.

It is NOT a reimplementation of anybody's model or runtime. The weights, the modelling code
(including `trust_remote_code` custom architectures) and the generation loop are the publisher's
own `transformers`; this process translates request and response fields. That distinction matters
for what a measurement taken through it is worth: the arm is the publisher's runtime, so it is
admissible evidence in a way our own decoder compared against itself never would be.

This is what makes two models measurable that were not:

  * `deepgrove/maple-preview` -- ollama refuses the artifact outright ("unsupported tensor
    blk.0.attn_k.weight size overflows"), and the publisher advertises transformers with
    `trust_remote_code=True` on CUDA with Triton and FlashAttention. That is the advertised route,
    so it is the one to run.
  * `JetBrains/Mellum2-12B-A2.5B-Thinking` -- ollama serves it fine, but JetBrains themselves ship
    it for transformers and vLLM. Measuring against the publisher's runtime is a stronger reference
    than measuring against a third-party re-packager.

PROTOCOL, transcribed from HttpGenerationClient.requestBody and LlamaCppGenerationParser:

  POST /completion
    {"prompt": str, "stream": true, "n_predict": int, "temperature": float, "top_k": int,
     "top_p": float, "seed": int, "n_threads": int, "repeat_penalty": float,
     "cache_prompt": false, "stop": [str]}

  Response: line-delimited JSON, one line per token, then a final line carrying the timings:
    {"content": "<piece>"}
    {"content": "", "stop": true,
     "timings": {"prompt_n": int, "prompt_ms": float, "predicted_n": int, "predicted_ms": float}}

  The parser takes time-to-first-token from the first event with non-empty `content`, so a token
  must be flushed as it is produced and metadata must never carry text. It raises if the stream
  ends without `stop` plus `timings`, and again if no output token arrived, so a failed generation
  cannot be mistaken for a fast one.

  GET /health -> 200 once the model is loaded. Readiness is the status, never the body: the pod
  runner probes for exactly 200 because a loading server that answers 503 with prose containing
  "ok" would otherwise read as ready.

USAGE
  pip install -U transformers accelerate torch
  python3 transformers-completion-server.py --model deepgrove/maple-preview --trust-remote-code \
      --device auto --port 8081

  Then point the harness at it, passing --artifact so the report carries an artifactSha256 and the
  comparison is admissible (RagBenchmarkCli refuses the arm otherwise):

  java -cp "$CP" com.integrallis.models.rag.RagBenchmarkCli \
      --framework plain-java --backend transformers \
      --backend-version "transformers@$(python3 -c 'import transformers;print(transformers.__version__)')" \
      --endpoint http://127.0.0.1:8081 --model deepgrove/maple-preview --artifact <path> ...

  --backend-version must name the real transformers version: the report is the only record of what
  produced the numbers, and "transformers" alone does not identify a generation loop.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Engine:
    """One model, one lock. Generations are serialised so a measurement is never contended."""

    def __init__(self, model_id, trust_remote_code, device, dtype):
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        kwargs = {"trust_remote_code": trust_remote_code}
        if device == "auto":
            kwargs["device_map"] = "auto"
        if dtype:
            kwargs["dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        if device not in ("auto",):
            self.model.to(device)
        self.device = next(self.model.parameters()).device

    def generate(self, request, write):
        """Greedy or sampled generation, flushing each token so TTFT is real."""
        prompt = request["prompt"]
        max_new = int(request.get("n_predict", 256))
        temperature = float(request.get("temperature", 0.0))
        top_k = int(request.get("top_k", 1))
        top_p = float(request.get("top_p", 1.0))
        seed = request.get("seed")
        repeat_penalty = float(request.get("repeat_penalty", 1.0))
        stops = request.get("stop") or []

        if seed is not None:
            torch.manual_seed(int(seed))

        # The harness sends the fully rendered prompt and compares arms on identical bytes, so the
        # chat template must NOT be applied here -- doing so would silently measure a different
        # prompt than the candidate arm received.
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        prompt_n = int(encoded["input_ids"].shape[-1])

        generated = []
        past = None
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        text_so_far = ""
        prefill_ms = None
        decode_started = None
        started = time.perf_counter()

        with torch.inference_mode():
            for step in range(max_new):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                logits = out.logits[:, -1, :].float()

                if prefill_ms is None:
                    # Everything up to the first produced logit is prefill, which is what
                    # timings.prompt_ms means to the parser.
                    prefill_ms = (time.perf_counter() - started) * 1000.0
                    decode_started = time.perf_counter()

                if repeat_penalty != 1.0 and generated:
                    for token in set(generated):
                        if logits[0, token] > 0:
                            logits[0, token] /= repeat_penalty
                        else:
                            logits[0, token] *= repeat_penalty

                if temperature <= 0.0 or top_k == 1:
                    next_token = int(torch.argmax(logits, dim=-1)[0])
                else:
                    scaled = logits / temperature
                    if top_k > 0:
                        kth = torch.topk(scaled, top_k, dim=-1).values[:, -1, None]
                        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
                    probabilities = torch.softmax(scaled, dim=-1)
                    if 0.0 < top_p < 1.0:
                        ordered, order = torch.sort(probabilities, descending=True, dim=-1)
                        cumulative = torch.cumsum(ordered, dim=-1)
                        ordered[cumulative - ordered > top_p] = 0.0
                        probabilities = torch.zeros_like(probabilities).scatter(
                            -1, order, ordered
                        )
                        probabilities /= probabilities.sum(dim=-1, keepdim=True)
                    next_token = int(torch.multinomial(probabilities, num_samples=1)[0, 0])

                generated.append(next_token)
                decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
                piece = decoded[len(text_so_far) :]
                text_so_far = decoded
                if piece:
                    write({"content": piece})

                if next_token == self.tokenizer.eos_token_id:
                    break
                if any(stop and stop in text_so_far for stop in stops):
                    break

                input_ids = torch.tensor([[next_token]], device=self.device)
                attention_mask = None

        predicted_ms = (time.perf_counter() - decode_started) * 1000.0 if decode_started else 0.0
        write(
            {
                "content": "",
                "stop": True,
                "timings": {
                    "prompt_n": prompt_n,
                    "prompt_ms": prefill_ms or 0.0,
                    "predicted_n": len(generated),
                    "predicted_ms": predicted_ms,
                },
            }
        )


def handler_for(engine, ready):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", ""):
                code = 200 if ready.is_set() else 503
                body = b'{"status":"ok"}' if code == 200 else b'{"status":"loading"}'
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            if self.path.rstrip("/") != "/completion":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def write(event):
                payload = (json.dumps(event) + "\n").encode("utf-8")
                self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

            try:
                with engine.lock:
                    engine.generate(request, write)
            except Exception as failure:  # surfaced as a stream without timings, which the parser rejects
                write({"content": "", "error": str(failure)})
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    ready = threading.Event()
    print(f"loading {args.model} (device={args.device})", flush=True)
    engine = Engine(args.model, args.trust_remote_code, args.device, args.dtype)
    ready.set()
    print(f"ready on {args.host}:{args.port}, device={engine.device}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), handler_for(engine, ready))
    server.serve_forever()


if __name__ == "__main__":
    main()
