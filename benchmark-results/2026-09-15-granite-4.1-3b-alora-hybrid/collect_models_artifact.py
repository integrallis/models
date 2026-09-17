#!/usr/bin/env python3
"""Write the component report's ``gates.modelsArtifact`` evidence for a Models release.

Downloads the jar and pom of every Models module the ModelJars component gate requires
(``REQUIRED_MODELS_MODULES`` in tools/component-evidence-gate.mjs) from Maven Central, hashes the
bytes actually received, and writes them in the gate's shape:

    {"pass": true, "version": V, "artifacts": [{"module", "coordinate",
      "jar": {"uri", "sha256", "sizeBytes"}, "pom": {"uri", "sha256", "sizeBytes"}}]}

The gate re-downloads every file and compares bytes, so this record only has to be honest, not
trusted. Pass the output to assemble_component_report.py --models-artifact.

    collect_models_artifact.py --version 0.3.42 --output models-artifact.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Callable

from assemble_component_report import MAVEN_CENTRAL, RELEASE_VERSION, REQUIRED_MODELS_MODULES, validate_models_artifact

Fetch = Callable[[str], bytes]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """The gate fetches with redirect: "error"; a URI that only works through a redirect is refused here too."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(f"{req.full_url} redirected ({code}) to {newurl}")


def central_fetch(uri: str) -> bytes:
    with urllib.request.build_opener(_NoRedirect).open(uri, timeout=120) as response:
        return response.read()


def collect(version: str, fetch: Fetch) -> dict:
    if RELEASE_VERSION.fullmatch(version) is None:
        raise ValueError(f"{version!r} is not a release version (major.minor.patch)")
    artifacts = []
    for module in REQUIRED_MODELS_MODULES:
        artifact = {"module": module, "coordinate": f"com.integrallis:{module}:{version}"}
        for kind in ("jar", "pom"):
            uri = f"{MAVEN_CENTRAL}/com/integrallis/{module}/{version}/{module}-{version}.{kind}"
            data = fetch(uri)
            if not data:
                raise ValueError(f"{uri} returned an empty body")
            artifact[kind] = {"uri": uri, "sha256": hashlib.sha256(data).hexdigest(), "sizeBytes": len(data)}
        artifacts.append(artifact)
    return validate_models_artifact({"pass": True, "version": version, "artifacts": artifacts})


def main(argv: list[str] | None = None, fetch: Fetch = central_fetch) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="Models release version, e.g. 0.3.42")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    gate = collect(args.version, fetch)
    args.output.write_text(json.dumps(gate, indent=2) + "\n")
    for artifact in gate["artifacts"]:
        print(f"{artifact['coordinate']}: jar {artifact['jar']['sizeBytes']} B, pom {artifact['pom']['sizeBytes']} B")


if __name__ == "__main__":
    main()
