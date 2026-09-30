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

Prompts must use characters in the recorded training vocabulary. Generation uses temperature 0.8 and the configured sliding context. Cached decoding is the default. To compare the reference path, add `--mode reference`.

The cache grows until the 96-character context is full. Subsequent tokens trigger a fresh prefill of the most recent window; this deliberately preserves the original experiment's sliding-window semantics. Long generation therefore does not keep a sustained cache speedup. A prompt longer than the context is cropped for inference, but the original prompt remains in the returned text.

Inspect model routing and run the tests:

```sh
python inspect_run.py
python -m unittest discover -s tests -v
python benchmark_inference.py
python verify_inference.py
```

The benchmark and inference audit refuse to replace existing outputs unless you pass `--overwrite`. This only replaces their own results, not the training artifacts. `verification.json` remains the historical training audit at commit `ba1e819`; use `runs/inference_verification.json` for the current inference code. The legacy `verify_run.py` rewrites the historical audit and is retained for the original experiment, not the recommended inference check.

The architecture classes can be reused from `model.py`. A model call returns `(logits, balancing_loss, assignment_fractions)`. The loss multiplier belongs to the training objective, not the model's forward calculation. `top_k=1` is allowed for inspection, but its normalized single mixture weight gives no task-loss gradient to the router; use the tested top-2 setting for this experiment.

For token-level integration without the character sampler:

```python
from model import load_model
from inference import InferenceEngine

engine = InferenceEngine(load_model("runs/moe"))
session = engine.start([30, 27, 25, 17, 27, 10, 0])  # ROMEO:\n
next_id = int(session.logits[0].numpy().argmax())
session.advance([[next_id]])
```

`engine.start` accepts a nonempty integer `[time]` or `[batch, time]` array. `session.advance` consumes exactly one integer token per batch member; it returns `[batch, vocabulary]` logits for the following token. Create a separate mutable session per request. The engine supports float32 and equal-length batches without padding, not a production multi-request scheduler. Direct `model.decode_step` calls require a same-model KV cache and raise at the context limit instead of truncating it silently.

Weights are local artifacts. Only load checkpoints you trust. No credentials, hosted model, or remote inference service is required.
