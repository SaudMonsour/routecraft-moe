# Command-line reference

Python 3.12 was used for the recorded run. Install `requirements.txt` in an isolated environment. `environment.lock.txt` records the complete executed environment.

Download the pinned corpus:

```sh
python prepare.py
```

Train all three architectures with identical sampled windows:

```sh
python train.py --steps 800 --batch 16 --context 96 --seed 42
```

The trainer refuses to overwrite an existing `runs/` directory. Preserve or move existing results before starting another experiment. Changing dimensions or training settings creates a different experiment; do not mix its metrics with the published run.

Generate from the included MoE checkpoint:

```sh
python generate.py --checkpoint runs/moe --prompt "ROMEO:" --characters 240 --seed 123
```

Prompts must use characters in the recorded training vocabulary. Generation uses temperature 0.8 and the configured sliding context. It recomputes attention for each new character rather than using a KV cache.

Inspect model routing and run the tests:

```sh
python inspect_run.py
python -m unittest discover -s tests -v
python verify_run.py
```

The architecture classes can be reused from `model.py`. A model call returns `(logits, balancing_loss, assignment_fractions)`. The loss multiplier belongs to the training objective, not the model's forward calculation. `top_k=1` is allowed for inspection, but its normalized single mixture weight gives no task-loss gradient to the router; use the tested top-2 setting for this experiment.

Weights are local artifacts. Only load checkpoints you trust. No credentials, hosted model, or remote inference service is required.
