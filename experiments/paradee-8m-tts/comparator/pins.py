"""Pinned artifact for every comparator script. Digests are the triage's; a mismatch is a hard stop."""
import hashlib, pathlib, urllib.request

REPO = "sahilmahendrakar/Paradee-8M-v1.2"
REVISION = "ed2934d4df8a4fd891e29584c5e1104649409765"
FILES = {
    "onnx/paradee_int8.onnx": "a6989796692ac173597e15d405c971989202d5c73cfad65f22e90bd7b6c97da1",
    "onnx/paradee.onnx": "8bac7b0bebef403cb22b54ead6bee423fa34a04fd91f73326b7f65cd9d0d25b9",
    "config.json": "002535cf8dd08a434cdddb13372da8a7b4844152bdf9b5b4e7342e12ec30bcfb",
}
VOICES = ["af_heart", "bf_isabella", "am_puck", "bm_fable"]
SAMPLE_RATE = 24000
CACHE = pathlib.Path(__file__).resolve().parent / ".artifacts"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def artifact(name: str) -> pathlib.Path:
    """Download once from the pinned revision and verify the digest every time."""
    target = CACHE / name.replace("/", "_")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", target)
    actual = sha256(target)
    if actual != FILES[name]:
        raise SystemExit(f"{name}: sha256 {actual} != pinned {FILES[name]}")
    return target


def load_corpus(path: str) -> dict:
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prepare_espeak() -> str:
    """Make misaki's eSpeak fallback work with the espeakng-loader wheel on this host.

    The loader's libespeak-ng (1.52.0, Intel macOS wheel) ignores the data path handed to
    espeak_Initialize and, when ESPEAK_DATA_PATH is set, checks that <path>/espeak-ng-data is a
    directory but then reads its files from <path> itself. A directory of symlinks to the wheel's
    data files that also contains a self-link named espeak-ng-data satisfies both reads. The data
    bytes are the wheel's own; only the directory they are read from changes. misaki sets an
    explicit data path at import, which this build ignores, so it is reset to None here after
    the import and before the publisher's code constructs the fallback.
    """
    import os
    import espeakng_loader
    data = pathlib.Path(espeakng_loader.get_data_path())
    farm = CACHE / "espeak-data"
    if not (farm / "espeak-ng-data").exists():
        farm.mkdir(parents=True, exist_ok=True)
        for entry in data.iterdir():
            link = farm / entry.name
            if not link.exists():
                link.symlink_to(entry)
        (farm / "espeak-ng-data").symlink_to(".")
    os.environ["ESPEAK_DATA_PATH"] = str(farm)
    from misaki import espeak  # noqa: F401  (sets the explicit path this build ignores)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_data_path(None)
    return str(farm)
