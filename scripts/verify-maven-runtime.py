#!/usr/bin/env python3
"""Check real Maven consumers of staged or published Models runtimes, independently of Gradle metadata."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repository', required=True)
parser.add_argument('--output', type=Path, default=Path('build/maven-runtime-smoke'))
args = parser.parse_args()
repository = args.repository
local_repository = None if repository.startswith('https://') else Path(repository).resolve()
repository_url = repository if local_repository is None else local_repository.as_uri()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
(out / 'results.json').unlink(missing_ok=True)
root = Path(__file__).resolve().parent.parent
version = re.search(r'^version\s*=\s*(\S+)', (root / 'gradle.properties').read_text(), re.M).group(1)
vectors_version = re.search(r'^vectorsVersion\s*=\s*(\S+)', (root / 'gradle.properties').read_text(), re.M).group(1)
vectors = {'com.integrallis:vectors-core': vectors_version}
jackson = {
    'com.fasterxml.jackson.core:jackson-core': '2.21.7',
    'com.fasterxml.jackson.core:jackson-annotations': '2.21',
    'com.fasterxml.jackson.core:jackson-databind': '2.21.7',
}
cases = {
    'backend-java': {**vectors, 'com.fasterxml.jackson.core:jackson-core': '2.21.7'},
    'models-router': vectors,
    'models-spring-ai': {**vectors, **jackson},
    'models-langchain4j': {**vectors, **jackson},
}
results = {}
# Each execution gets a fresh Maven cache so an earlier candidate with the same release GAV
# cannot conceal a broken POM. Each module is consumed alone to prevent one runtime from
# accidentally repairing another runtime's dependency graph through nearest-wins mediation.
with tempfile.TemporaryDirectory(prefix='models-maven-consumer-') as temporary:
    for artifact, expected in cases.items():
        if local_repository is not None:
            staged = local_repository / 'com/integrallis' / artifact / version / f'{artifact}-{version}.pom'
            if not staged.is_file():
                raise FileNotFoundError(staged)
        project = ET.Element('project', xmlns='http://maven.apache.org/POM/4.0.0')
        for key, value in [('modelVersion', '4.0.0'), ('groupId', 'audit'),
                           ('artifactId', artifact + '-consumer'), ('version', '1')]:
            ET.SubElement(project, key).text = value
        repo = ET.SubElement(ET.SubElement(project, 'repositories'), 'repository')
        ET.SubElement(repo, 'id').text = 'models-candidate'
        ET.SubElement(repo, 'url').text = repository_url
        dependency = ET.SubElement(ET.SubElement(project, 'dependencies'), 'dependency')
        for key, value in [('groupId', 'com.integrallis'), ('artifactId', artifact), ('version', version)]:
            ET.SubElement(dependency, key).text = value
        case = out / artifact
        case.mkdir(exist_ok=True)
        pom = case / 'pom.xml'
        ET.ElementTree(project).write(pom, encoding='utf-8', xml_declaration=True)
        tree = case / 'dependencies.txt'
        # Remove a previous result before execution; a failed Maven command must never reuse it.
        tree.unlink(missing_ok=True)
        command = ['mvn', '-B', '-f', str(pom), f'-Dmaven.repo.local={temporary}/repository',
                   'org.apache.maven.plugins:maven-dependency-plugin:3.8.1:tree',
                   f'-DoutputFile={tree}']
        with (case / 'maven.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        resolved = dict(re.findall(r'([\w.\-]+:[\w.\-]+):jar:([^:\s]+):', tree.read_text()))
        expected = {**expected, **{key: vectors_version for key in resolved
                                  if key.startswith('com.integrallis:vectors-')}}
        mismatches = {key: {'expected': value, 'actual': resolved.get(key)}
                      for key, value in expected.items() if resolved.get(key) != value}
        results[artifact] = {'expected': expected, 'resolved': resolved, 'mismatches': mismatches}
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        if mismatches:
            raise RuntimeError(f'{artifact}: unexpected Maven dependency versions: {mismatches}')
        print(f'{artifact}: {len(expected)} patched runtime dependencies verified', flush=True)
