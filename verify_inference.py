"""Audit inference changes without replacing the historical training audit."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from inference import InferenceEngine, generate_text
from model import load_model
from train import read_data, evaluate

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from case_ids(test)
        else:
            yield test.id()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace only the current inference audit")
    args = parser.parse_args()
    output = ROOT / "runs/inference_verification.json"
    if output.exists() and not args.overwrite:
        parser.error("inference audit exists; use --overwrite to replace it explicitly")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    original = json.loads((ROOT / "verification.json").read_text())
    changed = {"model.py", "generate.py", "README.md", "USAGE.md"}
    preserved = {}
    for relative, expected in original["artifact_sha256"].items():
        if relative not in changed:
            actual = sha256(ROOT / relative)
            if actual != expected:
                raise ValueError(f"published experiment artifact changed unexpectedly: {relative}")
            preserved[relative] = actual
    result = json.loads((ROOT / "runs/metrics.json").read_text())
    manifest, (_, _, test) = read_data(result["context"])
    benchmark = json.loads((ROOT / "runs/inference_benchmark.json").read_text())
    if benchmark["training_metrics_sha256"] != sha256(ROOT / "runs/metrics.json"):
        raise ValueError("benchmark refers to different training metrics")
    if benchmark["source_sha256"] != manifest["sha256"]:
        raise ValueError("benchmark refers to a different corpus")
    for name, scenario in benchmark["scenarios"].items():
        size = 16 if name == "growing-prefix" else 96
        np.testing.assert_array_equal(scenario["prompt_ids"], test[:size])
        np.testing.assert_array_equal(scenario["continuation_ids"], test[size:size+64])
    checks = {}
    for name, recorded in result["models"].items():
        engine = InferenceEngine(load_model(ROOT / "runs" / name))
        observed, losses = evaluate(engine.model, test, result["context"])
        nll_difference = abs(observed["cross_entropy_nats"] - recorded["test"]["cross_entropy_nats"])
        max_block_difference = float(np.max(np.abs(np.load(ROOT / "runs" / name / "test_block_losses.npy") - losses)))
        if nll_difference > 1e-6 or max_block_difference > 1e-5:
            raise ValueError(f"training metrics no longer replay: {name}")
        expected_sample = (ROOT / "runs" / name / "sample.txt").read_text().removesuffix("\n")
        sampled = {}
        for cached in (False, True):
            value = generate_text(engine, manifest["vocabulary"], "ROMEO:\n", count=240,
                                  seed=123, temperature=.8, cached=cached)
            if value != expected_sample:
                raise ValueError(f"published sample differs: {name}, cached={cached}")
            sampled["cached" if cached else "reference"] = True
        maximum_difference = 0.0
        for scenario, workload in benchmark["scenarios"].items():
            left = engine.start(workload["prompt_ids"])
            right = engine.start(workload["prompt_ids"], cached=False)
            for step in range(len(workload["continuation_ids"]) + 1):
                a, b = left.logits.numpy(), right.logits.numpy()
                np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)
                maximum_difference = max(maximum_difference, float(np.max(np.abs(a - b))))
                np.testing.assert_array_equal(a.argmax(-1), b.argmax(-1))
                if step < len(workload["continuation_ids"]):
                    token = [[workload["continuation_ids"][step]]]
                    left.advance(token)
                    right.advance(token)
            metrics = benchmark["models"][name][scenario]
            if metrics["cached_resets"] != left.cache_resets or metrics["final_kv_payload_bytes"] != left.cache_bytes():
                raise ValueError("cache reset or memory accounting differs")
            for mode in ("reference", "cached"):
                samples = metrics[mode]["request_latency_samples_ms"]
                if len(samples) != benchmark["measured_requests_per_mode"] or min(samples) <= 0:
                    raise ValueError("invalid timing observations")
                for percentile in (50, 95):
                    np.testing.assert_allclose(np.percentile(samples, percentile), metrics[mode][f"request_ms_p{percentile}"])
            ratio = metrics["reference"]["request_ms_p50"] / metrics["cached"]["request_ms_p50"]
            np.testing.assert_allclose(ratio, metrics["median_reference_over_cached_ratio"])
        counts = engine.trace_counts()
        if counts != {"full_prefix": 1, "prefill": 1, "decode_step": 1}:
            raise ValueError(f"unexpected inference retracing: {counts}")
        checks[name] = {"nll_difference": nll_difference, "max_block_nll_difference": max_block_difference,
                        "published_sample_matches": sampled, "max_absolute_cached_logit_difference": maximum_difference,
                        "greedy_predictions_match": True, "graph_traces": counts,
                        "parameters": engine.model.count_params()}
        print(json.dumps({"model": name, **checks[name]}), flush=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    ids = list(case_ids(suite))
    stream = io.StringIO()
    outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    if not outcome.wasSuccessful() or outcome.skipped:
        raise ValueError("architecture or inference test failed or was skipped")
    environment = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    if environment.strip() != (ROOT / "environment.lock.txt").read_text().strip():
        raise ValueError("environment differs from the published lock")
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and "__pycache__" not in path.parts
                   and ".git" not in path.parts and path.name not in {"input.txt", "metrics.partial.json", output.name})
    audit = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
             "checkpoint_parent_commit": benchmark["checkpoint_parent_commit"],
             "python": platform.python_version(), "tensorflow": tf.__version__,
             "source_sha256": manifest["sha256"], "checkpoint_replay": checks,
             "historical_training_audit_sha256": sha256(ROOT / "verification.json"),
             "preserved_original_artifacts": preserved,
             "artifact_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in files},
             "tests": {"passed": outcome.testsRun, "skipped": len(outcome.skipped), "test_ids": ids},
             "attribution": "AI-assisted inference implementation and automated verification for Saud Alotaibi. "
                            "No retraining, changed checkpoints, paid model calls, or replaced training results."}
    output.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
