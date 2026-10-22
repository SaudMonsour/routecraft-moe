# RouteCraft MoE

A small TensorFlow language model built to make sparse expert routing visible and testable.

## The question

Can a Transformer retain the quality of a larger feed-forward network while using only part of its expert weights for each token?

This project implements the model internals rather than calling an LLM API. There are no pretrained weights: the embeddings, attention projections, router, and expert networks all learn from Tiny Shakespeare. It is a character-level architecture experiment, not a general-purpose chatbot.

## Architecture

Two pre-normalized decoder blocks, four attention heads, a 64-dimensional residual stream, and a 96-character context. Each block uses rotary query/key positions and a causal attention mask. Its feed-forward sublayer contains four SwiGLU experts; a learned router chooses two for each token and normalizes their mixture weights.

```mermaid
flowchart TD
    A[Character embeddings] --> B[Pre-norm causal attention with RoPE]
    B --> C[Attention residual]
    C --> D[Pre-norm learned router]
    D --> E[Top-2 sparse expert dispatch]
    E --> F[Weighted expert outputs and residual]
    F --> G[Repeat decoder block]
    G --> H[Final norm and tied output projection]
```

Only assigned tokens enter an expert's matrix multiplications. No tokens are dropped, and there is no expert capacity cutoff. The auxiliary balancing loss discourages overloaded routing; it does not enforce uniform assignments. All expert weights remain stored in memory.

Attention, adjacent-pair RoPE, sparse dispatch, mixture aggregation, tied output projection, and the training loop are implemented in this repository. TensorFlow supplies tensor operations, automatic differentiation, dense layers, normalization, and the optimizer. This combines established techniques; it does not introduce a new MoE algorithm.

## Measured results

Training finished on **1 October 2026, Riyadh time**. Each architecture received exactly **1,228,800 sampled character targets**, using the same windows in the same order, seed 42, and 800 optimizer steps. Checkpoints were chosen on 128 fixed validation windows; final validation and test metrics cover 111,456 targets each.

| Model | Total parameters | Active per-token estimate | Validation NLL | Test perplexity | Test next-character accuracy |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Dense, active-matched | 135,872 | 135,872 | 1.8448 | 7.0968 | 43.65% |
| Dense, total-matched | 234,176 | 234,176 | **1.7944** | **6.9003** | **44.07%** |
| RouteCraft MoE | 234,688 | 136,384 | 1.8113 | 6.9610 | 43.52% |

The training-only unigram baseline reached test perplexity **28.4441**. Lower perplexity is better. NLL is cross-entropy in nats per character, without the auxiliary routing loss.

The larger dense model won on validation and test perplexity. The MoE model came close with about 42% fewer parameters involved per token, but that estimate is **not a FLOP count or a speedup**. On this CPU, median forward latency was 1.95 ms for MoE versus 1.77 ms for the total-matched dense model. Sparse dispatch has overhead at this scale.

This is one seed on a small corpus. Equal token budgets do not mean equal wall-clock compute, and these differences are not evidence of a statistically established advantage.

## Inference engineering: KV caching

The trained models now have a reusable inference engine, not just a training script. A prefill builds each block's key/value cache. Each subsequent step computes attention and expert routing for the new token only, with the correct rotary position offset. The weights and recorded training results have not changed.

On the published CPU checkpoints, a **16-character prompt followed by 64 teacher-forced updates** produced these whole-request medians:

| Model | Full-prefix reference | Cached path | Reference / cached |
| :--- | ---: | ---: | ---: |
| Dense, active-matched | 71.24 ms | 50.30 ms | 1.42× |
| Dense, total-matched | 81.61 ms | 61.03 ms | 1.34× |
| RouteCraft MoE | 87.44 ms | **60.64 ms** | **1.44×** |

Twenty trials per mode, alternating which mode runs first, follow three warmup requests. Timing includes prefill, Python orchestration, and materialized outputs; it excludes graph tracing, checkpoint loading, and sampling. The reference emits all prefix logits, while cached prefill emits the last position and cache tensors. This measures these two inference paths, not an isolated attention kernel or a universal speedup.

The important boundary is the **96-character context**. Simply removing an old KV entry would leave information from outside the window in deeper hidden states. Instead, the engine re-prefills the latest window when full. A 96-character prompt plus 64 updates therefore causes 64 resets: RouteCraft measured 149.47 ms for the reference and 139.91 ms for the cache path, **without reusing KV computation across those updates**. Short-prefix gains should not be extrapolated to long generation.

![Cached and reference request latency in two context regimes](figures/kv-cache-benchmark.png)

Across both benchmark workloads and all three checkpoints, greedy predictions matched; the largest absolute logit difference was `7.04e-6`. Cached and reference sampling also match the original 240-character samples with the recorded seed. All **15 architecture and inference tests** pass. Each engine graph traces once while sequence length and equal-length batch size vary.

KV payload at batch size one is 80 KiB after the growing-prefix request and 96 KiB at the context limit. These figures exclude weights, allocator overhead, graph storage, and temporary concatenation copies. There is no paged cache, ragged batching, GPU measurement, or production request scheduler.

## What the router learned

Both blocks use all four experts on held-out text. Assignment shares range from about 15% to 35%, rather than collapsing into one route. The category heatmap shows different routing patterns for spaces, capitalization, lowercase letters, and punctuation. These are character-group associations, not proof of semantic expert specialization.

`runs/routing_by_character.csv` contains the actual counts. An independent routing pass checks those counts against the recorded evaluation metrics. Every test token receives exactly two expert assignments in each block.

## Diagnostics

| Training progress | Held-out comparison |
| :---: | :---: |
| ![Validation learning curves](figures/learning-curves.png) | ![Test perplexity](figures/model-comparison.png) |

| Stored and active parameters | Observed CPU latency |
| :---: | :---: |
| ![Parameter comparison](figures/parameter-comparison.png) | ![CPU forward latency](figures/latency.png) |

| Expert usage | Routing by character group |
| :---: | :---: |
| ![Expert balance](figures/expert-balance.png) | ![Expert routing](figures/expert-routing.png) |

## Where it still fails

The checkpoints learn recognizable word fragments and dialogue formatting, but generated passages are often ungrammatical and inconsistent. They do not demonstrate factual reasoning or instruction following. The 96-character context limits longer dependencies; the highest-loss held-out windows are included in `runs/error_analysis.json` for inspection. Samples use the same prompt, temperature, and generation seed for every model, without choosing a flattering output.

The original training experiment's CPU timings cover a single 96-character forward pass without a KV cache: five warmup calls, then 30 measured calls with materialized output. They exclude tokenizer and sampling work and remain unchanged. The separate inference measurements above evaluate the added cache. Neither experiment includes distributed expert execution or a GPU benchmark.

## Data and split

[Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt), 1,115,394 characters. The contiguous split is 80% training, 10% validation, and 10% test. The 65-character vocabulary is learned from training text only, and no input/target window crosses a split boundary. Repeated phrases can still occur across plays; contiguous splitting is not a deduplication guarantee.

The source commit, full SHA-256, vocabulary, and exact boundaries are in `data/manifest.json`. The unmodified corpus is downloaded from that pinned source rather than redistributed here.

## Repository structure

- `model.py` — attention, RoPE, SwiGLU, sparse experts, decoder, and KV primitives.
- `train.py` — fixed-budget training, checkpoint selection, and evaluation.
- `inference.py` — compiled prefill/decode graphs and request-local cache state.
- `generate.py` — cached or reference local checkpoint inference.
- `benchmark_inference.py` — alternating latency trials and equivalence checks.
- `inspect_run.py` — routing counts, error windows, and figures.
- `verify_run.py` — checkpoint replay and artifact integrity checks.
- `verify_inference.py` — current-code replay, cache equivalence, and preserved-artifact audit.
- `tests/` — causality, gradients, sparse/reference equivalence, reloads, cache isolation, and window resets.
- `runs/` — trained checkpoints, measured results, training windows, and generated samples.
- `figures/` — plots derived from recorded results.
- `USAGE.md` — command-line reference.

`verification.json` is the immutable audit of the original training publication at commit `ba1e819`. Its source hashes describe that historical version, not the later inference code. `runs/inference_verification.json` records current-code hashes, checkpoint replay, all 15 tests, and which original artifacts remain byte-for-byte unchanged. The raw timing trials are in `runs/inference_benchmark.json`.

## References and provenance

- [Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html): sparse expert routing and load-balancing motivation. This project uses top-2 routing, not the paper's top-1 Switch design.
- [RoFormer](https://arxiv.org/abs/2104.09864): rotary position embeddings.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202): SwiGLU feed-forward networks.
- [TensorFlow function guide](https://www.tensorflow.org/guide/function): input signatures for shape-polymorphic inference graphs.

Built for Saud Alotaibi with AI-assisted implementation. Results come from executed local training and inference runs; no paid LLM calls or pretrained model outputs are used.
