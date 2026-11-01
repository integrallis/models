"""Package Central artifacts for the released-runtime repeat; never build native kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CENTRAL = 'https://repo.maven.apache.org/maven2/'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True)
    parser.add_argument('--dependency-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(['git','rev-parse','v'+args.version+'^{commit}'],cwd=REPO,text=True).strip()
    root = args.output
    root.mkdir(parents=True,exist_ok=False)
    for name in ['classes','lib','experiment','rust/target/release']:
        (root/name).mkdir(parents=True,exist_ok=True)
    records = []
    def fetch(group, artifact, version):
        stem = CENTRAL+group.replace('.','/')+'/'+artifact+'/'+version+'/'+artifact+'-'+version
        target = root/'lib'/f'{artifact}-{version}.jar'
        for suffix in ['.jar','.pom','.jar.asc','.pom.asc']:
            with urllib.request.urlopen(stem+suffix,timeout=120) as source:
                data = source.read()
            if not data:
                raise ValueError('empty Central artifact: '+stem+suffix)
            if suffix == '.jar':
                target.write_bytes(data)
            records.append({'url': stem+suffix, 'sha256': hashlib.sha256(data).hexdigest(), 'sizeBytes': len(data)})
        return target
    for module in ['models-api','backend-java','backend-native']:
        fetch('com.integrallis',module,args.version)
    # Resolve the same frozen supporting dependencies from Central, never copy a local candidate JAR.
    for artifact,group,version in [('vectors-core','com.integrallis','0.1.32'),
            ('jackson-databind','com.fasterxml.jackson.core','2.22.3'),
            ('jackson-core','com.fasterxml.jackson.core','2.22.3'),
            ('jackson-annotations','com.fasterxml.jackson.core','2.22'),('slf4j-api','org.slf4j','2.0.20')]:
        downloaded = fetch(group,artifact,version)
        expected = args.dependency_directory/downloaded.name
        if not expected.is_file() or sha(downloaded) != sha(expected):
            raise ValueError('supporting dependency differs from qualified candidate: '+artifact)
    native_path = root/'rust/target/release/libjmodels_kernels.so'
    resource = 'META-INF/models/native/linux-x86_64/'
    with zipfile.ZipFile(root/'lib'/f'backend-native-{args.version}.jar') as jar:
        native_path.write_bytes(jar.read(resource+'libjmodels_kernels.so'))
        metadata = jar.read(resource+'native.properties').decode()
        if f'sha256={sha(native_path)}' not in metadata:
            raise ValueError('published native checksum mismatch')
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ['.json','.jsonl','.py','.java','.sh','.lock','.in']:
            shutil.copyfile(path,root/'experiment'/path.name)
    subprocess.run(['javac','--add-modules','jdk.incubator.vector','-cp',str(root/'lib/*'),
        '-d',str(root/'classes'),str(HERE/'GlinerWorker.java'),str(HERE/'RerankerRegression.java'),
        str(HERE/'ReleasedGlinerSmoke.java')],check=True)
    receipt = {'kind':'central-release','version':args.version,'sourceRevision':revision,
        'artifacts':records,'nativeSha256':sha(native_path),'runtimeFiles':{p.name:sha(p) for p in (root/'lib').glob('*.jar')}}
    (root/'released-runtime.json').write_text(json.dumps(receipt,indent=2)+'\n')
    archive = root.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as output:
        for path in sorted(root.rglob('*')):
            if path.is_file():output.add(path,arcname=str(path.relative_to(root)))
    manifest = {'sourceRevision':revision,'sha256':sha(archive),'runtimeKind':'central-release',
        'releasedVersion':args.version,'catalogQualified':False,
        'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}}
    root.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'payload':str(archive),'manifest':str(root.with_suffix('.json'))}))

if __name__ == '__main__':
    main()
