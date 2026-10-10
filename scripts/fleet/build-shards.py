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


def qualification_precedent(catalog_dir):
    """modelId -> {workload, promptTemplate} for every QUALIFIED model in the catalog.

    Richer and more durable than one campaign's reports: the catalog records the template and
    workload that actually earned each verdict, across every campaign there has ever been. Only
    rows with ``qualified`` true are read -- `entries` also holds rejections, and what a rejected
    model ran with is not precedent for anything.

    Returns an empty mapping rather than raising when the catalog is not to hand, because
    precedent is advisory: the tool refuses either way, and a missing catalog should narrow the
    advice, not break the refusal.
    """
    if not catalog_dir:
        return {}
    path = pathlib.Path(catalog_dir).parent / "qualifications.json"
    if not path.exists():
        return {}
    try:
        entries = json.loads(path.read_text())["entries"]
    except (json.JSONDecodeError, KeyError, OSError):
        return {}
    out = {}
    for e in entries:
        if e.get("qualified") is not True:
            continue
        out[e["modelId"]] = {"workload": e.get("workload"),
                             "promptTemplate": e.get("promptTemplate")}
    return out


def precedent(catalog_by_id, prior, model_id, qualified=None):
    """What comparable models actually ran with, for a person to choose from.

    Two sources, counted separately so their weight is visible: models QUALIFIED in the catalog
    (what earned a verdict) and reports from the runs being read now (what was tried recently).
    """
    me = catalog_by_id.get(model_id, {})
    my_arch = me.get("architecture")
    my_caps = set(me.get("capabilities", []))
    my_dims = me.get("dimensions")
    tpl_q, wl_q = collections.Counter(), collections.Counter()
    tpl_r, wl_r = collections.Counter(), collections.Counter()
    # Identical dimensions mean the same base model at the same quantization -- a far sharper
    # signal than a shared architecture. nexus-science is byte-for-byte nexus-legal in every
    # dimension, and legal is already qualified, so what legal ran with is the precedent that
    # matters; architecture qwen2 alone would have pointed at the most common template across
    # fifteen unrelated models instead.
    same_base = collections.Counter()
    same_base_models = []

    for mid, info in (qualified or {}).items():
        other = catalog_by_id.get(mid)
        if not other:
            continue
        if other.get("architecture") == my_arch and info.get("promptTemplate"):
            tpl_q[info["promptTemplate"]] += 1
        if my_caps and (my_caps & set(other.get("capabilities", []))) and info.get("workload"):
            wl_q[info["workload"]] += 1
        if (my_dims and other.get("dimensions") == my_dims and mid != model_id
                and info.get("promptTemplate")):
            same_base[info["promptTemplate"]] += 1
            same_base_models.append((mid, info["promptTemplate"], info.get("workload")))

    for mid, rec in prior.items():
        other = catalog_by_id.get(mid)
        if not other:
            continue
        st = rec["settings"]
        if other.get("architecture") == my_arch and st.get("promptTemplate"):
            tpl_r[st["promptTemplate"]] += 1
        if my_caps and (my_caps & set(other.get("capabilities", []))) and st.get("workload"):
            wl_r[st["workload"]] += 1

    return {"architecture": my_arch,
            "templates": dict(tpl_q), "workloads": dict(wl_q),
            "templatesRecent": dict(tpl_r), "workloadsRecent": dict(wl_r),
            "sameBase": dict(same_base), "sameBaseModels": sorted(same_base_models)}


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


def build_jobs(ids, catalog_by_id, prior, mt_override, wl_override, missing,
               tpl_override=None, dt_override=None):
    """Jobs for the requested ids, or reasons in ``missing``.

    A model with a prior report has its settings copied. A model WITHOUT one -- a candidate that
    has never run here -- can only be built if the caller declares both of the fields that decide
    whether the measurement means anything, ``wl`` and ``tpl``. Declaring one and leaving the other
    to a default would be the same guess this tool exists not to make, so both are required
    together and the job is reported as declared rather than copied.
    """
    tpl_override = tpl_override or {}
    dt_override = dt_override or {}
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
            if model_id in wl_override and model_id in tpl_override:
                rec = {
                    "generatedAt": "", "source": "(declared on the command line)",
                    "backendVersion": None, "diagnostics": {},
                    "settings": {"workload": wl_override[model_id],
                                 "promptTemplate": tpl_override[model_id],
                                 "maxOutputTokens": mt_override.get(model_id, 256)},
                    "declared": True,
                }
            else:
                have = [f for f, d in (("wl", wl_override), ("tpl", tpl_override))
                        if model_id in d]
                need = [f for f in ("wl", "tpl") if f not in have]
                detail = "no prior report records its settings"
                if have:
                    detail += (f"; {' and '.join(have)} was declared but "
                               f"{' and '.join(need)} was not -- both are required together")
                missing.append((model_id, detail))
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
            dt = dt_override.get(model_id, decode_threads(rec))
            if dt is not None:
                job["dt"] = dt
            job["_declared"] = bool(rec.get("declared"))
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
    ap.add_argument("--tpl-override", nargs="*", default=[], metavar="id=template",
                    help="prompt template; for a model with no prior report this is REQUIRED "
                         "alongside --wl-override, and neither is guessed")
    ap.add_argument("--dt-override", nargs="*", default=[], metavar="id=N",
                    help="decode threads (-Dmodels.native.kernels.decodeThreads), not --threads")
    args = ap.parse_args(argv)

    ids = [ln.split("#")[0].strip() for ln in pathlib.Path(args.ids).read_text().splitlines()]
    ids = [i for i in ids if i]
    raw = read_json(args.catalog)
    models = raw["models"] if isinstance(raw, dict) and "models" in raw else raw
    catalog_by_id = {m["id"]: m for m in models}
    prior = index_prior_reports(args.reports)
    mt_override = {k: int(v) for k, v in parse_overrides(args.mt_override).items()}
    wl_override = parse_overrides(args.wl_override)
    tpl_override = parse_overrides(args.tpl_override)
    dt_override = {k: int(v) for k, v in parse_overrides(args.dt_override).items()}

    for k in list(mt_override) + list(wl_override) + list(tpl_override) + list(dt_override):
        if k not in ids:
            sys.exit(f"override names {k!r}, which is not in --ids")

    qualified = qualification_precedent(args.catalog)
    missing = []
    jobs = build_jobs(ids, catalog_by_id, prior, mt_override, wl_override, missing,
                      tpl_override, dt_override)

    if missing:
        print(f"REFUSING: {len(missing)} of {len(ids)} models cannot be built from recorded "
              f"evidence.\n")
        for model_id, why in missing:
            print(f"  {model_id}: {why}")
            p = precedent(catalog_by_id, prior, model_id, qualified)
            if p["sameBaseModels"]:
                print(f"      SAME BASE MODEL -- identical dimensions to "
                      f"{len(p['sameBaseModels'])} already-qualified model(s), which is the "
                      f"strongest precedent available:")
                for mid2, tpl2, wl2 in p["sameBaseModels"]:
                    print(f"          {mid2:<46} tpl={tpl2}  qualified on {wl2}")
            if any(p[k] for k in ("templates", "workloads", "templatesRecent",
                                  "workloadsRecent")):
                print(f"      architecture {p['architecture']} -- templates that have QUALIFIED: "
                      f"{p['templates'] or '(none)'}")
                if p["templatesRecent"]:
                    print(f"          templates tried in the reports read here: "
                          f"{p['templatesRecent']}")
                print(f"      its capabilities -- workloads that have QUALIFIED: "
                      f"{p['workloads'] or '(none)'}")
                if p["workloadsRecent"]:
                    print(f"          workloads tried in the reports read here: "
                          f"{p['workloadsRecent']}")
                print(f"      none of these is a safe default -- see this script's docstring. "
                      f"Declare --wl-override and --tpl-override together, or add a prior "
                      f"report.")
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
            for field, over in (("mt", mt_override), ("wl", wl_override),
                                ("tpl", tpl_override), ("dt", dt_override)):
                if j["id"] in over:
                    tag += f"  {field} OVERRIDDEN to {j.get(field, '-')}"
            print(f"    {j['id']:<52} tpl={j['tpl']:<18} mt={j['mt']:<5} "
                  f"dt={j.get('dt', 'pool')}{tag}")
            if j["_declared"]:
                print(f"        DECLARED on the command line -- no prior report exists for this "
                      f"model, so wl and tpl were chosen by a person, not copied")
            else:
                print(f"        settings from {j['_from']}")
                print(f"        prior backend {j['_priorBackend']}")
    print()
    declared = [j["id"] for group in shards.values() for j in group if j["_declared"]]
    print(f"{len(jobs)} jobs across {len(shards)} shards. Every tpl/wl/dt/mt above was copied from "
          f"a prior report for that same model, except where marked OVERRIDDEN or DECLARED.")
    if declared:
        print(f"{len(declared)} job(s) were DECLARED rather than copied, because no prior report "
              f"exists for them:")
        for d in declared:
            print(f"    {d}")
        print("Their verdicts are only comparable to a later run that holds the same declaration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
