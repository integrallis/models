"""Pinned Hugging Face Transformers reference server for controlled RAG comparisons."""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterable


def apply_stop_sequences(text: str, stop_sequences: Iterable[str]) -> tuple[str, bool]:
    """Returns text through the earliest configured stop sequence."""
    positions = [text.find(stop) for stop in stop_sequences if stop and stop in text]
    if not positions:
        return text, False
    return text[: min(positions)], True


def content_event(content: str) -> dict[str, Any]:
    return {"content": content}


def final_event(
    *, prompt_tokens: int, output_tokens: int, prompt_millis: float
) -> dict[str, Any]:
    return {
        "content": "",
        "stop": True,
        "timings": {
            "prompt_n": prompt_tokens,
            "predicted_n": output_tokens,
            "prompt_ms": prompt_millis,
        },
    }


def validate_greedy_controls(request: dict[str, Any]) -> None:
    """Rejects settings this intentionally narrow reference does not implement."""
    expected = {
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "repeat_penalty": 1,
        "cache_prompt": False,
    }
    mismatches = {
        name: request.get(name)
        for name, value in expected.items()
        if request.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "transformers reference supports only deterministic controls; "
            f"received incompatible values: {mismatches}"
        )


def model_loading_options(trust_remote_code: bool) -> dict[str, bool]:
    """Builds the explicit, local-only Hugging Face loading policy."""
    return {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
    }


class TransformersReference:
    """Greedy, cache-backed generation over one pinned local model directory."""

    def __init__(
        self,
        model_directory: Path,
        threads: int,
        context_length: int,
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not model_directory.is_dir():
            raise ValueError(f"model directory does not exist: {model_directory}")
        if threads < 1 or context_length < 1:
            raise ValueError("threads and context length must be positive")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        self._torch = torch
        loading_options = model_loading_options(trust_remote_code)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_directory, **loading_options
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_directory,
            dtype=torch.bfloat16,
            **loading_options,
        ).eval()
        self._context_length = context_length
        self._eos_ids = self._resolve_eos_ids()

    def _resolve_eos_ids(self) -> set[int]:
        eos = self._model.generation_config.eos_token_id
        if eos is None:
            return set()
        if isinstance(eos, int):
            return {eos}
        return {int(token) for token in eos}

    def generate(self, prompt: str, max_tokens: int, stop_sequences: tuple[str, ...]):
        if not prompt or max_tokens < 1:
            raise ValueError("prompt must be non-empty and n_predict must be positive")
        torch = self._torch
        encoded = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False, return_attention_mask=True
        )
        prompt_tokens = int(encoded.input_ids.shape[-1])
        if prompt_tokens + max_tokens > self._context_length:
            raise ValueError(
                f"request exceeds context length: {prompt_tokens} + {max_tokens}"
            )

        prefill_start = time.perf_counter_ns()
        with torch.inference_mode():
            result = self._model(**encoded, use_cache=True)
        prompt_millis = (time.perf_counter_ns() - prefill_start) / 1_000_000
        next_token = int(torch.argmax(result.logits[0, -1]).item())
        past = result.past_key_values
        generated: list[int] = []
        visible = ""

        for _ in range(max_tokens):
            if next_token in self._eos_ids:
                break
            generated.append(next_token)
            decoded = self._tokenizer.decode(
                generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            allowed, stopped = apply_stop_sequences(decoded, stop_sequences)
            delta = allowed[len(visible) :] if allowed.startswith(visible) else allowed
            visible = allowed
            if delta:
                yield content_event(delta)
            if stopped:
                break

            token = torch.tensor([[next_token]], dtype=torch.long)
            with torch.inference_mode():
                result = self._model(input_ids=token, past_key_values=past, use_cache=True)
            past = result.past_key_values
            next_token = int(torch.argmax(result.logits[0, -1]).item())

        yield final_event(
            prompt_tokens=prompt_tokens,
            output_tokens=len(generated),
            prompt_millis=prompt_millis,
        )


def handler(reference: TransformersReference):
    class CompletionHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ready"}\n')

        def do_POST(self) -> None:
            if self.path != "/completion":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                validate_greedy_controls(request)
                prompt = str(request["prompt"])
                max_tokens = int(request["n_predict"])
                stop = tuple(str(value) for value in request.get("stop", ()))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for event in reference.generate(prompt, max_tokens, stop):
                    payload = json.dumps(event, separators=(",", ":"))
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
            except Exception as failure:
                if not self.wfile.closed:
                    self.log_error("completion failed: %r", failure)
                self.close_connection = True

        def log_message(self, format: str, *args: Any) -> None:
            return

    return CompletionHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow custom Python model code already present in the pinned local snapshot",
    )
    args = parser.parse_args()
    reference = TransformersReference(
        args.model_dir,
        args.threads,
        args.context,
        trust_remote_code=args.trust_remote_code,
    )
    server = HTTPServer((args.host, args.port), handler(reference))
    print(f"transformers reference ready on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
