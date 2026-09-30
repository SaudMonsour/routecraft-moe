"""Measure the same teacher-forced requests with cached and reference decoding."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from inference import InferenceEngine
from model import load_model
from train import read_data

ROOT = Path(__file__).resolve().parent


def run_request(engine, prompt, continuation, cached):
    begin = time.perf_counter_ns()
    session = engine.start(prompt, cached=cached)
    # Synchronize every prediction, not just the last asynchronous operation.
    session.logits.numpy()
    for token in continuation:
        session.advance([[int(token)]]).numpy()
    elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
    return elapsed_ms, session


def check_equivalence(engine, prompt, continuation):
    cached = engine.start(prompt)
    reference = engine.start(prompt, cached=False)
    differences, agreement = [], []
    for step in range(len(continuation) + 1):
        left, right = cached.logits.numpy(), reference.logits.numpy()
        np.testing.assert_allclose(left, right, atol=2e-5, rtol=2e-5)
        differences.append(float(np.max(np.abs(left - right))))
        agreement.append(bool(np.array_equal(left.argmax(-1), right.argmax(-1))))
        if step < len(continuation):
            token = [[int(continuation[step])]]
            cached.advance(token)
            reference.advance(token)
    if not all(agreement):
        raise ValueError("cached and reference greedy decisions differ on the benchmark request")
    return {"max_absolute_logit_difference": max(differences),
            "all_greedy_predictions_match": True,
            "predictions_checked": len(differences),
            "cached_resets": cached.cache_resets,
            "final_kv_payload_bytes": cached.cache_bytes()}


def plot_results(result):
    labels = {"dense-active": "Dense active", "dense-total": "Dense total", "moe": "RouteCraft MoE"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for axis, scenario in zip(axes, ("growing-prefix", "full-window")):
        names = list(result["models"])
        positions = np.arange(len(names))
        for shift, mode, color, label in ((-.18, "reference", "#8a99ab", "Full-prefix reference"),
                                          (.18, "cached", "#167b76", "Cached path")):
            medians = [result["models"][name][scenario][mode]["request_ms_p50"] for name in names]
            bars = axis.bar(positions + shift, medians, .34, color=color, label=label)
            for bar, median in zip(bars, medians):
                axis.text(bar.get_x() + bar.get_width()/2, median + 2, f"{median:.1f}",
                          ha="center", va="bottom", fontsize=9)
        axis.set_xticks(positions, [labels[name] for name in names], rotation=12)
        axis.set_title("16-char prompt + 64 updates" if scenario == "growing-prefix"
                       else "96-char prompt + 64 window resets", fontsize=11)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Median request latency (ms) — lower is better")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_ylim(0, max(axis.get_ylim()[1] for axis in axes) * 1.12)
    fig.suptitle("KV caching helps only while the context has room", fontsize=14, weight="bold")
    fig.text(.5, .02, f"CPU · batch 1 · {result['measured_requests_per_mode']} alternating trials per mode · "
             "prefill and host overhead included · no sampling", ha="center", fontsize=9, color="#4a5565")
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(ROOT / "figures/kv-cache-benchmark.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace the inference benchmark, not training results")
    args = parser.parse_args()
    if args.trials < 5 or args.warmup < 1:
        parser.error("at least five trials and one warmup request are required")
    output = ROOT / "runs/inference_benchmark.json"
    if output.exists() and not args.overwrite:
        parser.error("inference benchmark exists; use --overwrite to replace that benchmark explicitly")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    original = json.loads((ROOT / "runs/metrics.json").read_text())
    context = original["context"]
    if context != 96:
        parser.error("these benchmark scenarios require the published 96-character checkpoints")
    manifest, (_, _, test) = read_data(context)
    scenarios = {"growing-prefix": (test[:16], test[16:80]),
                 "full-window": (test[:96], test[96:160])}
    result = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint_parent_commit": "ba1e819820b3f31821c14245e4921bc3111d7aa9",
              "source_sha256": manifest["sha256"],
              "training_metrics_sha256": hashlib.sha256((ROOT / "runs/metrics.json").read_bytes()).hexdigest(),
              "measured_requests_per_mode": args.trials, "warmup_requests_per_mode": args.warmup,
              "method": "Identical teacher-forced IDs; initial prediction plus 64 appended tokens. "
                        "Whole-request timing includes prefill, 64 updates, Python orchestration, NumPy window copies, "
                        "and materialized logits at every position. It excludes graph tracing, model loading and sampling. "
                        "Modes alternate first position on each trial. Reference calculates all prefix logits; "
                        "cached prefill returns last-position logits and KV tensors. Single process, batch 1, CPU.",
              "scenarios": {name: {"prompt_ids": prompt.tolist(), "continuation_ids": continuation.tolist(),
                                     "prompt_characters": len(prompt), "appended_characters": len(continuation)}
                            for name, (prompt, continuation) in scenarios.items()},
              "runtime": {"python": platform.python_version(), "tensorflow": tf.__version__,
                          "platform": platform.platform(), "intra_threads": 2, "inter_threads": 2,
                          "oneDNN": False, "devices": [str(device) for device in tf.config.list_physical_devices()]},
              "models": {}}
    for name in original["models"]:
        engine = InferenceEngine(load_model(ROOT / "runs" / name))
        measured = {}
        for scenario, (prompt, continuation) in scenarios.items():
            checks = check_equivalence(engine, prompt, continuation)
            for _ in range(args.warmup):
                for cached in (False, True):
                    run_request(engine, prompt, continuation, cached)
            samples = {"reference": [], "cached": []}
            for trial in range(args.trials):
                order = (False, True) if trial % 2 == 0 else (True, False)
                for cached in order:
                    duration, _ = run_request(engine, prompt, continuation, cached)
                    samples["cached" if cached else "reference"].append(duration)
            stats = {mode: {"request_ms_p50": float(np.median(values)),
                            "request_ms_p95": float(np.percentile(values, 95)),
                            "request_latency_samples_ms": values} for mode, values in samples.items()}
            measured[scenario] = {**checks, **stats,
                                  "median_reference_over_cached_ratio": stats["reference"]["request_ms_p50"] / stats["cached"]["request_ms_p50"]}
            print(json.dumps({"model": name, "scenario": scenario,
                              "reference_ms": stats["reference"]["request_ms_p50"],
                              "cached_ms": stats["cached"]["request_ms_p50"],
                              "ratio": measured[scenario]["median_reference_over_cached_ratio"], **checks}), flush=True)
        measured["graph_traces"] = engine.trace_counts()
        result["models"][name] = measured
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(result, indent=2) + "\n")
    plot_results(result)


if __name__ == "__main__":
    main()
