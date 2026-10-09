#!/usr/bin/env python3
"""Build ``fleet-shard-NNN.json`` job files from recorded evidence instead of by hand.

Every field in a job file decides what gets measured, and two of them silently decide whether the
measurement means anything at all. Both were got wrong by hand on 2026-10-09:

* ``wl`` is the workload corpus. A shard built with ``general`` for every model returned
  FAILED_MODEL_CONTRIBUTION_GATE for a math specialist and a translation model that were already
  qualified in the catalog on ``math`` and ``multilingual``. Re-running eurollm on ``multilingual``
  with nothing else changed returned QUALIFIED.
* ``mt`` is the output cap. At the old hard-coded 256, fin-r1 truncated 78% of its answers, so the
  contribution metric measured the cap rather than the model.

The obvious fix -- derive ``wl`` from the model's capabilities and ``tpl`` from its architecture --
does not survive contact with the data, which is why this tool does not do it. Measured against the
catalog on 2026-10-09: the ``medical-reasoning`` capability maps to workload ``general``, not
``healthcare``; ``math`` splits between ``general`` and ``math``; ``reasoning`` is mostly
``general``; and ``chat`` and ``text-generation`` appear against all eight workloads. Prompt
templates are no better -- architecture ``llama`` has qualified under eight different templates and
``qwen2`` under three. Deriving either field would be guessing with a script's authority.

So this tool copies, and refuses. For each requested model it reads the settings that a PRIOR RUN
recorded for that same model -- workload, prompt template, threads, output cap -- out of that run's
own candidate report, and reuses them exactly. Where no prior report exists it prints the precedent
it found for that architecture and that capability set and then fails, leaving the choice to a
person. An override is explicit per field and is reported in the output.

Reusing the recorded settings is also what makes a confirmation run meaningful: re-measuring a model
against a released build is only a comparison if everything except the build is held fixed.

Usage:
    scripts/fleet/build-shards.py --ids ids.txt --reports ./prior-reports --out-dir /tmp \
        --start-shard 640 [--mt-override id=2048] [--catalog ../model-jars/catalog/models.json]
"""

import argparse
import collections
import json
import pathlib
import sys

# A box loads one corpus, so a shard carries one workload. Mixing them was never a correctness
# problem, only a wasteful one -- but the per-shard STATUS file is also how a workload's results are
# read back, and one workload per shard keeps that legible.
MAX_JOBS_PER_SHARD = 8


def read_json(path):
    return json.loads(pathlib.Path(path).read_bytes())


def index_prior_reports(report_dirs):
    """id -> the settings a prior run recorded, from the newest report found for that id.

    A candidate report is ``<id>.json`` carrying ``settings`` and ``generatedAt``. ``.verdict.json``
    and ``.comparator.json`` are skipped: the comparator is the baseline arm and its settings are
    derived from the candidate's, so reading it could only ever confirm what the candidate said.
    """
    found = {}
    for d in report_dirs:
        for path in sorted(pathlib.Path(d).rglob("*.json")):
            name = path.name
            if name.endswith((".verdict.json", ".comparator.json")):
                continue
            try:
                report = read_json(path)
            except (json.JSONDecodeError, OSError):
                continue
            if "settings" not in report or "modelId" not in report:
                continue
            mid = report["modelId"]
            stamp = report.get("generatedAt", "")
            if mid not in found or stamp > found[mid]["generatedAt"]:
                found[mid] = {
                    "generatedAt": stamp,
                    "source": str(path),
                    "backendVersion": report.get("backendVersion"),
                    "settings": report["settings"],
                    "diagnostics": (report.get("backendDiagnostics") or {}).get(
                        "environment") or {},
                }
    return found


def precedent(catalog_by_id, prior, model_id):
    """What comparable models actually ran with, for a human to choose from."""
    me = catalog_by_id.get(model_id, {})
    my_arch = me.get("architecture")
    my_caps = set(me.get("capabilities", []))
    tpl = collections.Counter()
    wl = collections.Counter()
    for mid, rec in prior.items():
        other = catalog_by_id.get(mid)
        if not other:
            continue
        s = rec["settings"]
        if other.get("architecture") == my_arch:
            tpl[s.get("promptTemplate")] += 1
        if my_caps and (my_caps & set(other.get("capabilities", []))):
            wl[s.get("workload")] += 1
    return {"architecture": my_arch, "templates": dict(tpl), "workloads": dict(wl)}


def decode_threads(rec):
    """The job's ``dt``, which is NOT the report's ``settings.threads``.

    Two different knobs, and conflating them was the first bug this tool had. ``settings.threads``
    is the declared thread budget handed to BOTH arms as ``--threads``, because sameWorkload()
    compares it; on the fleet box that is 16. ``dt`` is ours only -- it becomes
    ``-Dmodels.native.kernels.decodeThreads`` and sets how many threads the decode matmul uses,
    measured best at 8 on a 16-SMT-thread box. The report records it under
    ``backendDiagnostics.environment.native-kernel-decode-threads``.

    Returns None when the run used the whole pool, so the job omits ``dt`` and inherits the
    worker's default rather than pinning a number the prior run did not pin.
    """
    env = rec.get("diagnostics") or {}
    # backendDiagnostics values arrive as strings. The job schema uses numbers, and the worker
    # interpolates the field into a -D flag, so a quoted "8" would work by accident while making
    # the job file a different shape from every other one. Coerced, and a non-numeric value is
    # treated as absent rather than passed through.
    def as_int(value):
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    dt = as_int(env.get("native-kernel-decode-threads"))
    pool = as_int(env.get("native-kernel-threads"))
    if dt is None:
        return None
    if pool is not None and dt == pool:
        return None
    return dt


def build_jobs(ids, catalog_by_id, prior, mt_override, wl_override, missing):
    jobs = []
    for model_id in ids:
        cat = catalog_by_id.get(model_id)
        if not cat:
            missing.append((model_id, "not in the catalog"))
            continue
        if cat.get("format") != "gguf":
            missing.append((model_id, f"format is {cat.get('format')}, the fleet worker loads GGUF"))
            continue
        rec = prior.get(model_id)
        if not rec:
            missing.append((model_id, "no prior report records its settings"))
            continue
        s = rec["settings"]
        for field in ("workload", "promptTemplate"):
            if not s.get(field):
                missing.append((model_id, f"prior report has no {field}"))
                break
        else:
            size_gb = round((cat.get("sizeBytes") or 0) / 1e9, 2)
            if size_gb <= 0:
                missing.append((model_id, "catalog records no sizeBytes, so disk cannot be sized"))
                continue
            job = {
                "id": model_id,
                "uri": cat["downloadUri"],
                "tpl": s["promptTemplate"],
                "gb": size_gb,
                "arch": cat.get("architecture"),
                "wl": wl_override.get(model_id, s["workload"]),
                "mt": mt_override.get(model_id, s.get("maxOutputTokens", 256)),
            }
            dt = decode_threads(rec)
            if dt is not None:
                job["dt"] = dt
            job["_from"] = rec["source"]
            job["_priorBackend"] = rec["backendVersion"]
            jobs.append(job)
    return jobs


def shard(jobs, start):
    """One workload per shard, smallest model first, at most MAX_JOBS_PER_SHARD each."""
    by_wl = collections.defaultdict(list)
    for job in jobs:
        by_wl[job["wl"]].append(job)
    out = {}
    n = start
    for wl in sorted(by_wl):
        group = sorted(by_wl[wl], key=lambda j: j["gb"])
        for i in range(0, len(group), MAX_JOBS_PER_SHARD):
            out[n] = group[i:i + MAX_JOBS_PER_SHARD]
            n += 1
    return out


def parse_overrides(pairs):
    out = {}
    for item in pairs or []:
        if "=" not in item:
            sys.exit(f"override must be id=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True,
                    help="file of model ids, one per line; # comments allowed")
    ap.add_argument("--reports", required=True, nargs="+",
                    help="directories of prior candidate reports to read settings from")
    ap.add_argument("--catalog", required=True, help="path to modeljars catalog/models.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start-shard", type=int, required=True)
    ap.add_argument("--mt-override", nargs="*", default=[], metavar="id=N")
    ap.add_argument("--wl-override", nargs="*", default=[], metavar="id=workload")
    args = ap.parse_args(argv)

    ids = [ln.split("#")[0].strip() for ln in pathlib.Path(args.ids).read_text().splitlines()]
    ids = [i for i in ids if i]
    raw = read_json(args.catalog)
    models = raw["models"] if isinstance(raw, dict) and "models" in raw else raw
    catalog_by_id = {m["id"]: m for m in models}
    prior = index_prior_reports(args.reports)
    mt_override = {k: int(v) for k, v in parse_overrides(args.mt_override).items()}
    wl_override = parse_overrides(args.wl_override)

    for k in list(mt_override) + list(wl_override):
        if k not in ids:
            sys.exit(f"override names {k!r}, which is not in --ids")

    missing = []
    jobs = build_jobs(ids, catalog_by_id, prior, mt_override, wl_override, missing)

    if missing:
        print(f"REFUSING: {len(missing)} of {len(ids)} models cannot be built from recorded "
              f"evidence.\n")
        for model_id, why in missing:
            print(f"  {model_id}: {why}")
            p = precedent(catalog_by_id, prior, model_id)
            if p["templates"] or p["workloads"]:
                print(f"      precedent for architecture {p['architecture']}: "
                      f"templates {p['templates'] or '(none)'}")
                print(f"      precedent for its capabilities: workloads "
                      f"{p['workloads'] or '(none)'}")
                print(f"      neither is a safe default -- see this script's docstring. Pass the "
                      f"choice explicitly or add a prior report.")
        print()
        return 1

    shards = shard(jobs, args.start_shard)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, group in sorted(shards.items()):
        path = out_dir / f"fleet-shard-{n}.json"
        if path.exists():
            print(f"REFUSING: {path} already exists. Shard numbers are immutable -- a number that "
                  f"has been launched names a payload and a result prefix. Pick a later "
                  f"--start-shard.")
            return 1
        clean = [{k: v for k, v in j.items() if not k.startswith("_")} for j in group]
        path.write_text(json.dumps(clean, indent=1) + "\n")
        wl = group[0]["wl"]
        total = round(sum(j["gb"] for j in group), 2)
        print(f"shard {n}  [{wl}]  {len(group)} models  {total} GB  -> {path}")
        for j in group:
            tag = ""
            if j["id"] in mt_override:
                tag += f"  mt OVERRIDDEN to {j['mt']}"
            if j["id"] in wl_override:
                tag += f"  wl OVERRIDDEN to {j['wl']}"
            print(f"    {j['id']:<52} tpl={j['tpl']:<18} mt={j['mt']:<5} "
                  f"dt={j.get('dt', 'pool')}{tag}")
            print(f"        settings from {j['_from']}")
            print(f"        prior backend {j['_priorBackend']}")
    print()
    print(f"{len(jobs)} jobs across {len(shards)} shards. Every tpl/wl/dt/mt above was copied from "
          f"a prior report for that same model, except where marked OVERRIDDEN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
