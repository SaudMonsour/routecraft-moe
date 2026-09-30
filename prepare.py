"""Download a pinned Tiny Shakespeare source and preserve contiguous splits."""
import hashlib
import json
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
URL = f"https://raw.githubusercontent.com/karpathy/char-rnn/{SOURCE_COMMIT}/data/tinyshakespeare/input.txt"


def main():
    root = Path(__file__).resolve().parent / "data"
    root.mkdir(exist_ok=True)
    raw = urllib.request.urlopen(URL, timeout=60).read()
    text = raw.decode("utf-8")
    (root / "input.txt").write_bytes(raw)
    # Learn the vocabulary on training text only; never from validation/test.
    boundaries = [0, int(.8 * len(text)), int(.9 * len(text)), len(text)]
    vocab = sorted(set(text[:boundaries[1]]))
    if set(text) - set(vocab):
        raise ValueError("Held-out text contains unseen characters")
    manifest = {"source": URL, "source_commit": SOURCE_COMMIT,
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "characters": len(text), "vocabulary": vocab,
                "split_offsets": dict(zip(["train_start", "validation_start", "test_start", "end"], boundaries)),
                "split_policy": "Contiguous 80/10/10; no window crosses a split boundary.",
                "redistribution": "Original Shakespeare text is public domain; curated corpus provenance is retained."}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "vocabulary"}, indent=2))


if __name__ == "__main__":
    main()
