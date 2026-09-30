"""A small decoder with explicit causal attention and sparse top-2 experts."""
from dataclasses import dataclass, asdict
import tensorflow as tf


@dataclass
class Config:
    vocab_size: int = 65
    context: int = 96
    width: int = 64
    heads: int = 4
    layers: int = 2
    hidden: int = 128
    experts: int = 4
    top_k: int = 2
    architecture: str = "moe"
    balance_weight: float = 0.01

    def __post_init__(self):
        if self.architecture not in ("dense", "moe"):
            raise ValueError("architecture must be dense or moe")
        if min(self.vocab_size, self.context, self.width, self.heads, self.layers, self.hidden, self.experts) < 1:
            raise ValueError("dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must divide into even-dimensional heads")
        if not 1 <= self.top_k <= self.experts or self.balance_weight < 0:
            raise ValueError("invalid routing configuration")


def rotary(x):
    """Adjacent-pair RoPE on [batch, heads, time, head_width]."""
    dimension = tf.shape(x)[-1]
    positions = tf.cast(tf.range(tf.shape(x)[-2]), x.dtype)
    frequencies = tf.pow(tf.cast(10000.0, x.dtype), -tf.cast(tf.range(0, dimension, 2), x.dtype) / tf.cast(dimension, x.dtype))
    angles = positions[:, None] * frequencies[None, :]
    cosine, sine = tf.cos(angles), tf.sin(angles)
    even, odd = x[..., 0::2], x[..., 1::2]
    pairs = tf.stack([even * cosine - odd * sine, even * sine + odd * cosine], axis=-1)
    return tf.reshape(pairs, tf.shape(x))


class CausalAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.heads, self.head_width = config.heads, config.width // config.heads
        self.qkv = tf.keras.layers.Dense(3 * config.width, use_bias=False)
        self.output_projection = tf.keras.layers.Dense(config.width, use_bias=False)

    def call(self, x):
        batch, time = tf.shape(x)[0], tf.shape(x)[1]
        qkv = tf.reshape(self.qkv(x), [batch, time, 3, self.heads, self.head_width])
        q, k, v = tf.unstack(tf.transpose(qkv, [2, 0, 3, 1, 4]), axis=0)
        q, k = rotary(q), rotary(k)
        scores = tf.matmul(q, k, transpose_b=True) * (self.head_width ** -0.5)
        mask = tf.linalg.band_part(tf.ones([time, time], dtype=tf.bool), -1, 0)
        scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))
        values = tf.matmul(tf.nn.softmax(scores, axis=-1), v)
        values = tf.reshape(tf.transpose(values, [0, 2, 1, 3]), [batch, time, self.heads * self.head_width])
        return self.output_projection(values)


class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, width, hidden, **kwargs):
        super().__init__(**kwargs)
        self.up = tf.keras.layers.Dense(hidden, use_bias=False)
        self.gate = tf.keras.layers.Dense(hidden, use_bias=False)
        self.down = tf.keras.layers.Dense(width, use_bias=False)

    def call(self, x):
        return self.down(self.up(x) * tf.nn.silu(self.gate(x)))


class SparseExperts(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.width, self.top_k, self.expert_count = config.width, config.top_k, config.experts
        self.router = tf.keras.layers.Dense(config.experts, use_bias=False)
        self.expert_layers = [SwiGLU(config.width, config.hidden, name=f"expert_{i}") for i in range(config.experts)]

    def routing(self, x):
        flat = tf.reshape(x, [-1, self.width])
        probabilities = tf.nn.softmax(self.router(flat), axis=-1)
        weights, indices = tf.math.top_k(probabilities, k=self.top_k)
        weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
        return flat, probabilities, weights, indices

    def call(self, x):
        flat, probabilities, weights, indices = self.routing(x)
        output = tf.zeros_like(flat)
        # Each expert sees only its assigned tokens. No capacity truncation or token drops.
        for expert_id, expert in enumerate(self.expert_layers):
            assignments = tf.where(tf.equal(indices, expert_id))
            token_ids = assignments[:, 0]
            selected = tf.gather(flat, token_ids)
            routed = expert(selected) * tf.gather_nd(weights, assignments)[:, None]
            output = tf.tensor_scatter_nd_add(output, token_ids[:, None], routed)
        fractions = tf.reduce_mean(tf.one_hot(indices, self.expert_count), axis=[0, 1])
        importance = tf.reduce_mean(probabilities, axis=0)
        # Top-k adaptation: fractions sum to one across all assignments, not tokens.
        balance = self.expert_count * tf.reduce_sum(tf.stop_gradient(fractions) * importance)
        return tf.reshape(output, tf.shape(x)), balance, fractions


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.is_moe = config.architecture == "moe"
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attention = CausalAttention(config)
        self.ffn = SparseExperts(config) if self.is_moe else SwiGLU(config.width, config.hidden)

    def call(self, x):
        x = x + self.attention(self.norm1(x))
        if self.is_moe:
            delta, balance, fractions = self.ffn(self.norm2(x))
        else:
            delta = self.ffn(self.norm2(x))
            balance, fractions = tf.constant(0.0), tf.zeros([1])
        return x + delta, balance, fractions


class LanguageModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = tf.keras.layers.Embedding(config.vocab_size, config.width)
        self.blocks = [DecoderBlock(config, name=f"block_{i}") for i in range(config.layers)]
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, token_ids, training=False):
        tf.debugging.assert_less_equal(tf.shape(token_ids)[1], self.config.context)
        x = self.embedding(token_ids)
        balances, fractions = [], []
        for block in self.blocks:
            x, balance, fraction = block(x)
            balances.append(balance)
            fractions.append(fraction)
        x = self.final_norm(x)
        # Tied token embedding / output projection.
        logits = tf.einsum("btd,vd->btv", x, self.embedding.embeddings)
        return logits, tf.reduce_mean(balances), tf.stack(fractions)

    def parameter_accounting(self):
        total = self.count_params()
        if self.config.architecture == "dense":
            return {"total": total, "active_per_token_estimate": total}
        expert_parameters = sum(expert.count_params() for block in self.blocks for expert in block.ffn.expert_layers)
        active = total - expert_parameters + expert_parameters * self.config.top_k // self.config.experts
        return {"total": total, "active_per_token_estimate": active,
                "note": "Parameter involvement estimate, not FLOPs, memory usage, or a speed claim."}


def load_model(directory):
    import json
    from pathlib import Path
    directory = Path(directory)
    config = Config(**json.loads((directory / "config.json").read_text()))
    model = LanguageModel(config)
    model(tf.zeros([1, 1], dtype=tf.int32))
    model.load_weights(directory / "model.weights.h5")
    return model
