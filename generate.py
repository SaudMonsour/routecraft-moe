"""Generate text from a locally trained checkpoint."""
import argparse
import json
from pathlib import Path
from model import load_model
from train import generate, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/moe")
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--characters", type=int, default=240)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if not args.prompt or args.characters < 1:
        parser.error("prompt must be nonempty and characters positive")
    vocabulary = json.loads((ROOT / "data/manifest.json").read_text())["vocabulary"]
    if set(args.prompt) - set(vocabulary):
        parser.error("prompt has characters outside the training vocabulary")
    print(generate(load_model(args.checkpoint), vocabulary, args.prompt, args.characters, args.seed))


if __name__ == "__main__":
    main()
