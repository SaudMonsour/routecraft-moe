import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
import tensorflow as tf
from model import Config, LanguageModel, SparseExperts, rotary, load_model


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(42)
        self.config = Config(vocab_size=12, context=8, width=16, heads=2, layers=1, hidden=16)

    def test_causality(self):
        for architecture in ("dense", "moe"):
            config = Config(**dict(asdict(self.config), architecture=architecture))
            model = LanguageModel(config)
            a = model(tf.constant([[1, 2, 3, 4, 5]]))[0].numpy()
            b = model(tf.constant([[1, 2, 3, 9, 10]]))[0].numpy()
            np.testing.assert_allclose(a[:, :3], b[:, :3], atol=1e-6)

    def test_sparse_matches_dense_reference(self):
        layer = SparseExperts(self.config)
        x = tf.random.normal([2, 8, 16])
        sparse, balance, fractions = layer(x)
        flat, _, weights, indices = layer.routing(x)
        reference = tf.zeros_like(flat)
        for i, expert in enumerate(layer.expert_layers):
            gates = tf.reduce_sum(tf.where(indices == i, weights, 0), axis=-1)
            reference += expert(flat) * gates[:, None]
        np.testing.assert_allclose(sparse.numpy().reshape(-1, 16), reference.numpy(), atol=1e-6)
        self.assertAlmostEqual(float(tf.reduce_sum(fractions)), 1.0, places=6)
        self.assertTrue(np.isfinite(float(balance)))

    def test_router_has_gradient(self):
        layer = SparseExperts(self.config)
        x = tf.random.normal([2, 8, 16])
        with tf.GradientTape() as tape:
            output, balance, _ = layer(x)
            loss = tf.reduce_sum(output ** 2) + .01 * balance
        gradient = tape.gradient(loss, layer.router.trainable_variables)[0]
        self.assertTrue(np.all(np.isfinite(gradient.numpy())))
        self.assertGreater(float(tf.reduce_sum(tf.abs(gradient))), 0)

    def test_rotary_preserves_norm(self):
        x = tf.random.normal([2, 2, 8, 8])
        np.testing.assert_allclose(tf.reduce_sum(x * x, -1), tf.reduce_sum(rotary(x) ** 2, -1), rtol=1e-5)

    def test_checkpoint_round_trip(self):
        model = LanguageModel(self.config)
        x = tf.constant([[1, 2, 3]])
        before = model(x)[0].numpy()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text(json.dumps(asdict(self.config)))
            model.save_weights(directory / "model.weights.h5")
            after = load_model(directory)(x)[0].numpy()
        np.testing.assert_allclose(before, after, atol=1e-7)

    def test_context_limit(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            LanguageModel(self.config)(tf.zeros([1, 9], tf.int32))

    def test_invalid_config(self):
        for kwargs in ({"width": 15}, {"top_k": 5}, {"architecture": "unknown"}):
            with self.assertRaises(ValueError):
                Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
