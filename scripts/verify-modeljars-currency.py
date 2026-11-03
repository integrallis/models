#!/usr/bin/env python3
"""Fails when the documented ModelJars version is not the one Maven Central serves.

verifyDocumentation already keeps every document consistent with the modeljarsVersion property, so
the five pages that interpolate {modeljars-version} can never disagree with each other. What nothing
checked is whether that property is current: it is another project's release, so no module depends
on it, no lockfile pins it, and no build resolves it.

It drifted twice in one day. 0.1.65 was documented against a published 0.1.75 -- ten releases, and
0.1.65 predates the ONNX intake and the speech loader, so a reader following the docs got a runtime
that could not load models the catalog advertises. It was corrected to 0.1.75, and 0.1.76 shipped an
hour later, leaving it stale again.

Maven Central is the authority, for the reason ModelJars' own verify-site-version.mjs gives: a
source tag does not prove a reader can resolve the dependency the documentation shows.

Network failure is not a pass. Unreachable Central exits non-zero rather than waving the check
through, which is how a check of this shape first shipped doing nothing.

    python3 scripts/verify-modeljars-currency.py [root] [released-version]

An explicit released version skips the fetch, which is what the mutation check uses so that proving
this guard can fail does not depend on the network. ModelJars' verify-site-version.mjs takes the
same argument for the same reason.
"""
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

METADATA = "https://repo1.maven.org/maven2/org/modeljars/modeljars/maven-metadata.xml"
PROPERTY = re.compile(r"^modeljarsVersion\s*=\s*(\d+\.\d+\.\d+)\s*$", re.MULTILINE)


def documented(root: Path) -> str:
    text = (root / "gradle.properties").read_text()
    match = PROPERTY.search(text)
    if not match:
        raise SystemExit("gradle.properties declares no concrete modeljarsVersion")
    return match.group(1)


def released(url: str = METADATA) -> str:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as failure:
        raise SystemExit(f"Cannot reach Maven Central to check ModelJars currency: {failure}")
    match = re.search(r"<release>([^<]+)</release>", body)
    if not match:
        raise SystemExit("Maven Central metadata carries no <release> for org.modeljars:modeljars")
    return match.group(1)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    declared = documented(root)
    current = sys.argv[2] if len(sys.argv) > 2 else released()
    if declared != current:
        print(
            f"The documentation advertises ModelJars {declared} but Maven Central serves {current}.\n"
            f"Set modeljarsVersion = {current} in gradle.properties and run "
            f"./gradlew verifyDocumentation, which names every other file that carries it.",
            file=sys.stderr,
        )
        return 1
    print(f"ModelJars currency: documentation and Maven Central agree on {declared}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
