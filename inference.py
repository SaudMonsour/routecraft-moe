"""Reusable, request-local decoding with exact sliding-window cache resets."""
import numpy as np
import tensorflow as tf


def _token_array(value, vocabulary_size):
    value = np.asarray(value)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("token IDs must be a nonempty [batch, time] array")
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError("token IDs must be integers")
    if np.any(value < 0) or np.any(value >= vocabulary_size):
        raise ValueError("token ID outside vocabulary")
    return value.astype(np.int32, copy=False)


class InferenceEngine:
    """One loaded model, three shape-polymorphic TensorFlow graphs.

    Start a separate DecodeSession for each request. Equal-length batches are
    supported; padding, mixed precision, and variable-length batches are not.
    Neither cache growth nor prompt length creates a new graph trace.
    """
    def __init__(self, model):
        self.model = model
        config = model.config
        if not model.built:
            model(tf.zeros([1, 1], tf.int32))
        if model.compute_dtype != "float32":
            raise ValueError("the tested inference engine requires float32")
        prefix = tf.TensorSpec([None, None], tf.int32)
        token = tf.TensorSpec([None, 1], tf.int32)
        kv = tf.TensorSpec([None, config.heads, None, config.width // config.heads], tf.float32)
        cache_signature = tuple((kv, kv) for _ in model.blocks)
        self.full_prefix = tf.function(
            lambda ids: model(ids, training=False)[0][:, -1], input_signature=[prefix])
        self.prefill = tf.function(model.prefill, input_signature=[prefix])
        self.decode_step = tf.function(model.decode_step, input_signature=[token, cache_signature])

    def start(self, token_ids, cached=True):
        ids = _token_array(token_ids, self.model.config.vocab_size)
        return DecodeSession(self, ids[:, -self.model.config.context:], cached)

    def trace_counts(self):
        return {name: function.experimental_get_tracing_count()
                for name, function in (("full_prefix", self.full_prefix),
                                       ("prefill", self.prefill), ("decode_step", self.decode_step))}


class DecodeSession:
    """Mutable state for one request, not a shared or thread-safe cache.

    `logits` predicts the next token. `advance` consumes an externally supplied
    token for each batch member, then updates logits. It does not sample.
    """
    def __init__(self, engine, window, cached):
        self.engine = engine
        self.window = window.copy()
        self.cached = bool(cached)
        self.cache_resets = 0
        self.caches = None
        if self.cached:
            self.logits, self.caches = engine.prefill(self.window)
        else:
            self.logits = engine.full_prefix(self.window)

    def advance(self, token_ids):
        ids = _token_array(token_ids, self.engine.model.config.vocab_size)
        if ids.shape != (len(self.window), 1):
            raise ValueError("advance needs one token for each existing batch member")
        context = self.engine.model.config.context
        was_full = self.window.shape[1] == context
        window = np.concatenate([self.window, ids], axis=1)[:, -context:]
        if not self.cached:
            logits, caches = self.engine.full_prefix(window), None
        elif was_full:
            # Dropping only the oldest KV entry would retain information from
            # outside the window and would not match full-prefix decoding.
            logits, caches = self.engine.prefill(window)
        else:
            logits, caches = self.engine.decode_step(ids, self.caches)
        self.window, self.logits, self.caches = window, logits, caches
        if self.cached and was_full:
            self.cache_resets += 1
        return self.logits

    def cache_bytes(self):
        """KV tensor payload only, excluding weights, graphs, allocator and copies."""
        if self.caches is None:
            return 0
        return sum(int(tf.size(tensor)) * tensor.dtype.size
                   for pair in self.caches for tensor in pair)


def generate_text(engine, vocabulary, prompt, count=240, seed=123, temperature=.8, cached=True):
    """Sample using the original experiment's RNG and probability calculation."""
    if not prompt or count < 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("nonempty prompt, nonnegative count and positive temperature required")
    if len(vocabulary) != engine.model.config.vocab_size or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("vocabulary must match the checkpoint")
    lookup = {character: i for i, character in enumerate(vocabulary)}
    if set(prompt) - set(lookup):
        raise ValueError("prompt has characters outside the training vocabulary")
    ids = [lookup[character] for character in prompt]
    if count == 0:
        return prompt
    session = engine.start(ids, cached=cached)
    rng = np.random.default_rng(seed)
    for step in range(count):
        logits = session.logits[0].numpy() / temperature
        logits -= logits.max()
        probability = np.exp(logits) / np.exp(logits).sum()
        token = int(rng.choice(len(vocabulary), p=probability))
        ids.append(token)
        if step + 1 < count:
            session.advance([[token]])
    return "".join(vocabulary[i] for i in ids)
