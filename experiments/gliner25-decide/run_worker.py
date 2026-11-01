"""Bounded JSONL process driver: preserve readiness, errors, every case and wall time."""
import argparse
import json
import os
import platform
import queue
from pathlib import Path
import subprocess
import threading
import time

from common import read_cases


def run(command, cases, output, timeout=180):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.stderr').open('w') as errors, output.open('w') as results:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=errors, text=True, bufsize=1)
        lines = queue.Queue()
        def read():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)
        threading.Thread(target=read, daemon=True).start()
        def response():
            try:
                line = lines.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError('worker response timed out') from None
            if line is None:
                raise RuntimeError('worker exited before response')
            return json.loads(line)
        try:
            ready = response()
            if ready.get('type') != 'ready' or ready.get('device') != ['cpu']:
                raise ValueError('worker did not attest CPU readiness')
            ready['driverHostname'] = platform.node()
            ready['driverCpuAffinity'] = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
            results.write(json.dumps(ready) + '\n'); results.flush()
            for case in cases:
                start = time.perf_counter()
                process.stdin.write(json.dumps(case, ensure_ascii=False) + '\n')
                process.stdin.flush()
                result = response()
                if result.get('id') != case['id']:
                    raise ValueError('response id mismatch')
                result['wallMs'] = (time.perf_counter() - start) * 1000
                status = Path(f'/proc/{process.pid}/status')
                if status.exists():
                    for line in status.read_text().splitlines():
                        if line.startswith('VmHWM:'):
                            result['peakRssKiB'] = int(line.split()[1])
                results.write(json.dumps(result) + '\n'); results.flush()
            process.stdin.close()
            if process.wait(timeout=30) != 0:
                raise RuntimeError('worker failed on shutdown')
        finally:
            if process.poll() is None:
                process.kill(); process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[0] == '--' else args.command
    run(command, read_cases(args.cases), args.output)
