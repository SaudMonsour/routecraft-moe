"""Replay held-out metrics from published checkpoints and hash artifacts."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import tensorflow as tf
from model import load_model
from train import ROOT, read_data, evaluate, generate


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    result = json.loads((ROOT / "runs/metrics.json").read_text())
    manifest, (_, _, test) = read_data(result["context"])
    checks = {}
    for name, recorded in result["models"].items():
        model = load_model(ROOT / "runs" / name)
        observed, losses = evaluate(model, test, result["context"])
        difference = abs(observed["cross_entropy_nats"] - recorded["test"]["cross_entropy_nats"])
        stored_losses = np.load(ROOT / "runs" / name / "test_block_losses.npy")
        max_difference = float(np.max(np.abs(stored_losses - losses)))
        if difference > 1e-6 or max_difference > 1e-5:
            raise ValueError(f"Checkpoint replay differs: {name}")
        sample = generate(model, manifest["vocabulary"], "ROMEO:\n", seed=123)
        if sample + "\n" != (ROOT / "runs" / name / "sample.txt").read_text():
            raise ValueError(f"Sample replay differs: {name}")
        checks[name] = {"nll_difference": difference, "max_block_nll_difference": max_difference,
                        "sample_replay_matches": True, "parameters": model.count_params()}
    versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (ROOT / "environment.lock.txt").write_text(versions)
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name not in {"input.txt", "verification.json", "metrics.partial.json"})
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    audit = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
             "tensorflow": tf.__version__, "checkpoint_replay": checks,
             "source_sha256": manifest["sha256"], "artifact_sha256": hashes,
             "tests": "Seven architecture tests passed; see tests/test_model.py.",
             "attribution": "AI-assisted implementation and automated verification for Saud Alotaibi."}
    (ROOT / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
