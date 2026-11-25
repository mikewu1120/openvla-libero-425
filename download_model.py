"""Download the pinned, official LIBERO-Spatial checkpoint into this project."""
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "openvla/openvla-7b-finetuned-libero-spatial"
REVISION = "962318cec55ac10993ff0f5f43eda9a270b4c873"
ROOT = Path(__file__).resolve().parent
DEST = ROOT / "models" / "openvla-7b-finetuned-libero-spatial"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(entry):
    name = entry["path"]
    target = DEST / name
    expected = entry.get("lfs", {}).get("oid")
    if target.exists() and target.stat().st_size == entry["size"]:
        if not expected or sha256(target) == expected:
            print(f"Already verified: {name}", flush=True)
            return
    temporary = target.with_suffix(target.suffix + ".partial")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true"
    print(f"Downloading {name}: {entry['size'] / 1e9:.3f} GB", flush=True)
    # Bounded HTTP ranges avoid stalled multi-GB responses and allow resuming.
    offset = temporary.stat().st_size if temporary.exists() else 0
    with temporary.open("ab") as output:
        while offset < entry["size"]:
            end = min(offset + 32 * 1024 * 1024, entry["size"]) - 1
            request = urllib.request.Request(url + f"&part={offset}", headers={"Range": f"bytes={offset}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206:
                    raise ValueError(f"Server did not honor byte range: {name}")
                chunk = response.read()
            if len(chunk) != end - offset + 1:
                raise ValueError(f"Incomplete byte range: {name}")
            output.write(chunk)
            output.flush()
            offset += len(chunk)
            if offset // (512 * 1024 * 1024) != (offset - len(chunk)) // (512 * 1024 * 1024):
                print(f"{name}: {offset / 1e9:.2f}/{entry['size'] / 1e9:.2f} GB", flush=True)
    if temporary.stat().st_size != entry["size"]:
        raise ValueError(f"Incomplete download: {name}")
    if expected:
        if sha256(temporary) != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
    temporary.replace(target)
    print(f"Verified: {name}", flush=True)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}", timeout=30) as response:
        entries = json.load(response)
    files = [e for e in entries if e["type"] == "file" and e["path"].endswith((".json", ".model", ".safetensors"))]
    (DEST / "download_manifest.json").write_text(json.dumps({"repo": REPO, "revision": REVISION, "files": files}, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, files))
    print("Checkpoint download complete.", flush=True)
