"""Fixed-budget architecture experiment; test data is evaluated only at the end."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from model import Config, LanguageModel

ROOT = Path(__file__).resolve().parent


def read_data(context):
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    raw = (ROOT / "data/input.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("Corpus hash differs from manifest")
    text = raw.decode("utf-8")
    lookup = {c: i for i, c in enumerate(manifest["vocabulary"])}
    offsets = list(manifest["split_offsets"].values())
    pieces = [np.array([lookup[c] for c in text[a:b]], dtype=np.int32) for a, b in zip(offsets[:-1], offsets[1:])]
    if min(map(len, pieces)) <= context:
        raise ValueError("Split shorter than context")
    return manifest, pieces


def windows(piece, context):
    starts = np.arange(0, len(piece) - context, context)
    indices = starts[:, None] + np.arange(context)[None, :]
    return piece[indices], piece[indices + 1]


def ce(labels, logits):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)


def evaluate(model, piece, context, limit=None):
    x, y = windows(piece, context)
    if limit:
        x, y = x[:limit], y[:limit]
    losses, accuracies, routes = [], [], []
    @tf.function
    def forward(inputs):
        return model(inputs, training=False)
    for start in range(0, len(x), 16):
        logits, _, routing = forward(x[start:start + 16])
        losses.extend(tf.reduce_mean(ce(y[start:start + 16], logits), axis=1).numpy().tolist())
        accuracies.extend(np.mean(np.argmax(logits.numpy(), axis=-1) == y[start:start + 16], axis=1).tolist())
        routes.append((len(x[start:start + 16]), routing.numpy()))
    weights = sum(count for count, _ in routes)
    mean_routes = sum(count * route for count, route in routes) / weights
    return {"cross_entropy_nats": float(np.mean(losses)), "perplexity": float(np.exp(np.mean(losses))),
            "next_character_accuracy": float(np.mean(accuracies)), "evaluated_tokens": int(len(x) * context),
            "blocks": len(x), "routing_assignment_fractions": mean_routes.tolist()}, losses


def generate(model, vocabulary, prompt, count=240, seed=123):
    lookup = {c: i for i, c in enumerate(vocabulary)}
    ids = [lookup[c] for c in prompt]
    rng = np.random.default_rng(seed)
    @tf.function
    def forward(x):
        return model(x, training=False)[0]
    for _ in range(count):
        logits = forward(np.array([ids[-model.config.context:]], dtype=np.int32))[0, -1].numpy() / .8
        logits -= logits.max()
        probability = np.exp(logits) / np.exp(logits).sum()
        ids.append(int(rng.choice(len(vocabulary), p=probability)))
    return "".join(vocabulary[i] for i in ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--context", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.context) < 1:
        parser.error("steps, batch and context must be positive")
    output = ROOT / "runs"
    if output.exists():
        parser.error("runs/ already exists; preserve or move the previous experiment before a new run")
    output.mkdir()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    manifest, (train, validation, test) = read_data(args.context)
    rng = np.random.default_rng(args.seed)
    # Identical training windows, in identical order, for all architectures.
    schedule = rng.integers(0, len(train) - args.context, size=(args.steps, args.batch))
    np.save(output / "training_starts.npy", schedule)
    configs = [("dense-active", "dense", 256), ("dense-total", "dense", 512), ("moe", "moe", 128)]
    metrics, histories = {}, []
    started_at = datetime.now(timezone.utc).isoformat()
    for name, architecture, hidden in configs:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        config = Config(vocab_size=len(manifest["vocabulary"]), context=args.context, architecture=architecture, hidden=hidden)
        model = LanguageModel(config)
        model(tf.zeros([1, args.context], dtype=tf.int32))
        optimizer = tf.keras.optimizers.Adam(learning_rate=.001, global_clipnorm=1.0)
        optimizer.build(model.trainable_variables)
        directory = output / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits, balance, _ = model(x, training=True)
                nll = tf.reduce_mean(ce(y, logits))
                objective = nll + config.balance_weight * balance
            gradients = tape.gradient(objective, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return nll, balance
        best, best_step = math.inf, 0
        begun = time.perf_counter()
        for step, starts in enumerate(schedule, 1):
            ix = starts[:, None] + np.arange(args.context)[None, :]
            nll, balance = train_step(train[ix], train[ix + 1])
            if step == 1 or step % 100 == 0 or step == args.steps:
                # Fixed 128 validation blocks; never consult the test set for selection.
                val, _ = evaluate(model, validation, args.context, limit=128)
                row = {"model": name, "step": step, "training_nll": float(nll), "balance_loss": float(balance),
                       "validation_nll": val["cross_entropy_nats"], "elapsed_seconds": time.perf_counter() - begun}
                histories.append(row)
                print(json.dumps(row), flush=True)
                if row["validation_nll"] < best:
                    best, best_step = row["validation_nll"], step
                    model.save_weights(directory / "model.weights.h5")
        train_seconds = time.perf_counter() - begun
        model.load_weights(directory / "model.weights.h5")
        val, _ = evaluate(model, validation, args.context)
        final, block_losses = evaluate(model, test, args.context)
        # These are dispersion estimates over correlated contiguous blocks, not IID confidence intervals.
        np.save(directory / "test_block_losses.npy", np.array(block_losses))
        x, _ = windows(test, args.context)
        @tf.function
        def benchmark(inputs):
            return model(inputs, training=False)[0]
        for _ in range(5):
            benchmark(x[:1]).numpy()
        durations = []
        for _ in range(30):
            begin = time.perf_counter()
            benchmark(x[:1]).numpy()
            durations.append((time.perf_counter() - begin) * 1000)
        sample = generate(model, manifest["vocabulary"], "ROMEO:\n", seed=123)
        (directory / "sample.txt").write_text(sample + "\n")
        metrics[name] = {"parameters": model.parameter_accounting(), "selected_step": best_step,
                         "training_seconds_including_validation_and_saving": train_seconds,
                         "training_tokens": args.steps * args.batch * args.context,
                         "validation": val, "test": final,
                         "forward_latency_ms_p50": float(np.median(durations)), "forward_latency_ms_p95": float(np.percentile(durations, 95)),
                         "benchmark": "CPU, one 96-character window, 5 warmup + 30 measured forwards, materialized output; no KV cache.",
                         "forward_latency_samples_ms": durations}
        (output / "metrics.partial.json").write_text(json.dumps(metrics, indent=2) + "\n")
    frequencies = np.bincount(train, minlength=len(manifest["vocabulary"])).astype(float) + 1
    probabilities = frequencies / frequencies.sum()
    _, test_y = windows(test, args.context)
    unigram_loss = float(-np.log(probabilities[test_y]).mean())
    winner = min(metrics, key=lambda key: metrics[key]["validation"]["cross_entropy_nats"])
    result = {"started_at_utc": started_at, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "seed": args.seed, "steps": args.steps, "batch": args.batch, "context": args.context,
              "source_sha256": manifest["sha256"], "models": metrics,
              "unigram_baseline": {"test_nll": unigram_loss, "test_perplexity": math.exp(unigram_loss)},
              "selected_by_full_validation_nll": winner,
              "limitations": ["One seed; small corpus and short context.", "Character-level model, not a general-purpose LLM.",
                              "Equal sampled-token budgets, not equal wall-clock compute.", "CPU sparse dispatch has overhead; no assumed speedup.",
                              "Split boundaries do not guarantee absence of repeated phrases across Shakespeare plays."],
              "runtime": {"python": platform.python_version(), "tensorflow": tf.__version__, "platform": platform.platform(),
                          "intra_threads": 2, "inter_threads": 2, "oneDNN": False, "devices": [str(x) for x in tf.config.list_physical_devices()]}}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(histories[0]))
        writer.writeheader()
        writer.writerows(histories)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
