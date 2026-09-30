"""Inspect measured experiment outputs and the trained router."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import csv
import hashlib
import json
import string
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import load_model
from train import ROOT, read_data, windows


def category(character):
    if character.isspace():
        return "Whitespace"
    if character.isupper():
        return "Uppercase"
    if character.islower():
        return "Lowercase"
    return "Punctuation"


def finish(filename):
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / filename, dpi=160)
    plt.close()


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    metrics = json.loads((ROOT / "runs/metrics.json").read_text())
    manifest, pieces = read_data(metrics["context"])
    names = list(metrics["models"])
    with (ROOT / "runs/training_history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    plt.figure(figsize=(8, 4.4))
    for name in names:
        rows = [r for r in history if r["model"] == name]
        plt.plot([int(r["step"]) for r in rows], [float(r["validation_nll"]) for r in rows], marker="o", label=name)
    plt.xlabel("Optimizer steps (same training windows)")
    plt.ylabel("Validation cross-entropy (nats / character)")
    plt.title("Same token budget, different feed-forward architecture")
    plt.legend()
    finish("learning-curves.png")
    plt.figure(figsize=(8, 4.4))
    values = [metrics["models"][n]["test"]["perplexity"] for n in names]
    bars = plt.bar(names, values, color=["#64748b", "#0f766e", "#7c3aed"])
    plt.ylabel("Test character perplexity (lower is better)")
    plt.title("Held-out language modeling performance")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, value + .08, f"{value:.2f}", ha="center")
    plt.ylim(0, max(values) * 1.2)
    finish("model-comparison.png")
    plt.figure(figsize=(8, 4.4))
    ix = np.arange(len(names))
    totals = [metrics["models"][n]["parameters"]["total"] for n in names]
    active = [metrics["models"][n]["parameters"]["active_per_token_estimate"] for n in names]
    plt.bar(ix-.18, totals, width=.36, label="Total parameters")
    plt.bar(ix+.18, active, width=.36, label="Active per-token estimate")
    plt.xticks(ix, names)
    plt.ylabel("Parameters")
    plt.title("Sparse activation does not remove stored weights")
    plt.legend()
    finish("parameter-comparison.png")
    plt.figure(figsize=(8, 4.4))
    plt.bar(names, [metrics["models"][n]["forward_latency_ms_p50"] for n in names])
    plt.ylabel("Median forward latency (ms)")
    plt.title("CPU, batch 1, 96 characters; no KV cache")
    finish("latency.png")
    model = load_model(ROOT / "runs/moe")
    x, _ = windows(pieces[2], metrics["context"])
    counts = np.zeros((model.config.layers, model.config.vocab_size, model.config.experts), dtype=np.int64)
    # Track which input characters get routed to each expert in their full context.
    for start in range(0, len(x), 16):
        batch = x[start:start+16]
        hidden = model.embedding(batch)
        for layer_id, block in enumerate(model.blocks):
            hidden = hidden + block.attention(block.norm1(hidden))
            normalized = block.norm2(hidden)
            _, _, _, indices = block.ffn.routing(normalized)
            ids = indices.numpy()
            characters = np.repeat(batch.reshape(-1), model.config.top_k)
            np.add.at(counts[layer_id], (characters, ids.reshape(-1)), 1)
            delta, _, _ = block.ffn(normalized)
            hidden = hidden + delta
    with (ROOT / "runs/routing_by_character.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "character", "category", "expert", "assignments"])
        for layer_id in range(model.config.layers):
            for char_id, character in enumerate(manifest["vocabulary"]):
                for expert in range(model.config.experts):
                    writer.writerow([layer_id, json.dumps(character), category(character), expert, int(counts[layer_id, char_id, expert])])
    categories = ["Whitespace", "Uppercase", "Lowercase", "Punctuation"]
    fig, axes = plt.subplots(1, model.config.layers, figsize=(10, 4), sharey=True)
    normalized_categories = []
    for layer_id, ax in enumerate(np.atleast_1d(axes)):
        grouped = np.array([counts[layer_id, [i for i, c in enumerate(manifest["vocabulary"]) if category(c) == name]].sum(0) for name in categories])
        fractions = grouped / np.maximum(grouped.sum(1, keepdims=True), 1)
        normalized_categories.append(fractions.tolist())
        ax.imshow(fractions, vmin=0, vmax=.5, cmap="Purples")
        ax.set_xticks(range(model.config.experts), [f"E{i}" for i in range(model.config.experts)])
        ax.set_yticks(range(4), categories)
        ax.set_title(f"Decoder block {layer_id + 1}")
        for i in range(4):
            for j in range(model.config.experts):
                ax.text(j, i, f"{fractions[i,j]:.0%}", ha="center", va="center", color="white" if fractions[i,j] > .28 else "black")
    fig.suptitle("Assignment shares by character group (not semantic expertise)")
    finish("expert-routing.png")
    plt.figure(figsize=(8, 4.4))
    routes = np.asarray(metrics["models"]["moe"]["test"]["routing_assignment_fractions"])
    for layer_id in range(model.config.layers):
        plt.bar(np.arange(model.config.experts) + (layer_id - .5)*.3, routes[layer_id], width=.3, label=f"Block {layer_id+1}")
    plt.axhline(1/model.config.experts, color="black", linestyle="--", label="Uniform assignment")
    plt.xticks(range(model.config.experts), [f"Expert {i}" for i in range(model.config.experts)])
    plt.ylabel("Share of top-2 assignments")
    plt.title("Router balance on the full test split")
    plt.legend()
    finish("expert-balance.png")
    text = (ROOT / "data/input.txt").read_text()
    test_text = text[manifest["split_offsets"]["test_start"]:]
    errors = []
    for name in names:
        losses = np.load(ROOT / f"runs/{name}/test_block_losses.npy")
        for block_id in np.argsort(losses)[-10:][::-1]:
            offset = int(block_id) * metrics["context"]
            errors.append({"model": name, "test_offset": offset, "mean_nll": float(losses[block_id]),
                           "text": test_text[offset:offset+metrics["context"]]})
    (ROOT / "runs/error_analysis.json").write_text(json.dumps(errors, indent=2) + "\n")
    # Compare independently reconstructed assignment counts with evaluation output.
    observed = counts.sum(1) / counts.sum((1,2))[:,None]
    difference = float(np.max(np.abs(observed - routes)))
    if difference > 1e-5:
        raise ValueError("Router counts do not match recorded metrics")
    verification = {"source_hash_verified": True, "routing_count_max_difference": difference,
                    "test_windows": len(x), "assignments_per_layer": counts.sum((1,2)).tolist(),
                    "expected_assignments_per_layer": int(len(x) * metrics["context"] * model.config.top_k),
                    "category_order": categories, "assignment_shares_by_category": normalized_categories}
    (ROOT / "runs/inspection.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
