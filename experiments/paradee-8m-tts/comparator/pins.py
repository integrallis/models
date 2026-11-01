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
