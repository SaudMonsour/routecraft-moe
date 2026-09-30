"""Generate text from a locally trained checkpoint."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import argparse
import json
from pathlib import Path
import tensorflow as tf
from model import load_model
from inference import InferenceEngine, generate_text

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/moe")
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--characters", type=int, default=240)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mode", choices=("cached", "reference"), default="cached")
    args = parser.parse_args()
    if not args.prompt or args.characters < 1:
        parser.error("prompt must be nonempty and characters positive")
    vocabulary = json.loads((ROOT / "data/manifest.json").read_text())["vocabulary"]
    if set(args.prompt) - set(vocabulary):
        parser.error("prompt has characters outside the training vocabulary")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    engine = InferenceEngine(load_model(args.checkpoint))
    print(generate_text(engine, vocabulary, args.prompt, args.characters, args.seed,
                        cached=args.mode == "cached"))


if __name__ == "__main__":
    main()
