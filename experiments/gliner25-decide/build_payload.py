"""Assemble a committed runtime and its exact dependencies; no weights or inference."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent


def main():
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip():
        raise ValueError('commit tracked changes before assembling an immutable payload')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    root=REPO/'build/gliner-runtime'/revision
    root.mkdir(parents=True,exist_ok=False)
    for folder in ['classes','lib','rust','experiment']:(root/folder).mkdir()
    for jar in (REPO/'build/gliner-qualification/lib').glob('*.jar'):
        shutil.copyfile(jar,root/'lib'/jar.name)
    version=next(line.split('=',1)[1].strip() for line in (REPO/'gradle.properties').read_text().splitlines() if line.startswith('version ='))
    for module in ['models-api','backend-java','backend-native']:
        jar=REPO/module/'build/libs'/f'{module}-{version}.jar'
        if not jar.is_file():raise ValueError('missing built runtime jar: '+str(jar))
        for old in (root/'lib').glob(module+'-*.jar'):old.unlink()
        shutil.copyfile(jar,root/'lib'/jar.name)
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ['.json','.jsonl','.py','.java','.sh','.lock','.in']:
            shutil.copyfile(path,root/'experiment'/path.name)
    rust=REPO/'backend-native/src/main/rust/model-kernels'
    for name in ['Cargo.toml','Cargo.lock','LICENSE-BOOST-SLEEF.txt']:
        shutil.copyfile(rust/name,root/'rust'/name)
    shutil.copytree(rust/'src',root/'rust/src')
    subprocess.run(['javac','--add-modules','jdk.incubator.vector','-cp',str(root/'lib/*'),
        '-d',str(root/'classes'),str(HERE/'GlinerWorker.java'),str(HERE/'RerankerRegression.java')],check=True)
    archive=root.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as output:
        for path in sorted(root.rglob('*')):
            if path.is_file():output.add(path,arcname=str(path.relative_to(root)))
    def sha(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    manifest=dict(sourceRevision=revision,sha256=sha(archive),catalogQualified=False,
        files={str(path.relative_to(root)):sha(path) for path in root.rglob('*') if path.is_file()})
    root.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(payload=str(archive),manifest=str(root.with_suffix('.json')))))


if __name__=='__main__':main()
