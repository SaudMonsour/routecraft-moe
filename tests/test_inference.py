import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import unittest
import numpy as np
import tensorflow as tf
from inference import InferenceEngine, generate_text
from model import Config, LanguageModel, rotary


class InferenceTests(unittest.TestCase):
    def make_engine(self, architecture="moe"):
        tf.keras.utils.set_random_seed(42)
        return InferenceEngine(LanguageModel(Config(vocab_size=12, context=8,
                               width=16, heads=2, layers=2, hidden=16, architecture=architecture)))

    def test_rotary_offset_matches_full_sequence(self):
        x = tf.random.normal([2, 2, 8, 8])
        np.testing.assert_allclose(rotary(x)[:, :, 5:], rotary(x[:, :, 5:], offset=5), atol=1e-7)

    def test_prefill_and_incremental_match_full_prefix(self):
        for architecture in ("dense", "moe"):
            with self.subTest(architecture=architecture):
                engine = self.make_engine(architecture)
                ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], np.int32)
                session = engine.start(ids[:, :2])
                np.testing.assert_allclose(session.logits, engine.full_prefix(ids[:, :2]), atol=2e-6)
                for end in range(3, 9):
                    np.testing.assert_allclose(session.advance(ids[:, end-1:end]),
                                               engine.full_prefix(ids[:, :end]), atol=2e-6)
                self.assertEqual(session.cache_resets, 0)
                self.assertEqual(session.cache_bytes(), 2 * 2 * 2 * 8 * 16 * 4)

    def test_full_context_resets_match_reference(self):
        for architecture in ("dense", "moe"):
            engine = self.make_engine(architecture)
            prompt = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]], np.int32)
            cached, reference = engine.start(prompt), engine.start(prompt, cached=False)
            np.testing.assert_allclose(cached.logits, reference.logits, atol=2e-6)
            for token in (10, 11, 1, 2):
                np.testing.assert_allclose(cached.advance([[token]]), reference.advance([[token]]), atol=2e-6)
                self.assertEqual(cached.window.shape[1], 8)
            self.assertEqual(cached.cache_resets, 4)
            self.assertEqual(reference.cache_bytes(), 0)

    def test_new_requests_do_not_reuse_old_cache(self):
        engine = self.make_engine()
        first, second = engine.start([1, 2]), engine.start([7, 8, 9])
        second_before = second.logits.numpy().copy()
        first.advance([[3]])
        np.testing.assert_array_equal(second.logits.numpy(), second_before)
        np.testing.assert_allclose(second.logits, engine.full_prefix([[7, 8, 9]]), atol=2e-6)

    def test_cache_growth_and_batches_do_not_retrace(self):
        engine = self.make_engine()
        for batch in (1, 2):
            session = engine.start(np.ones([batch, 2], np.int32))
            for _ in range(7):
                session.advance(np.ones([batch, 1], np.int32))
            engine.full_prefix(np.ones([batch, 3], np.int32))
        self.assertEqual(engine.trace_counts(), {"prefill": 1, "decode_step": 1, "full_prefix": 1})

    def test_invalid_ids_and_step_shapes(self):
        engine = self.make_engine()
        for ids in ([], [12], [-1], [1.5], np.empty((0, 3), np.int32), np.zeros((1, 1, 1), np.int32)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                engine.start(ids)
        session = engine.start([1, 2])
        for ids in ([[1, 2]], [[1], [2]], [[12]]):
            with self.assertRaises(ValueError):
                session.advance(ids)
        np.testing.assert_array_equal(session.window, [[1, 2]])

    def test_invalid_low_level_cache_is_rejected(self):
        engine = self.make_engine()
        _, caches = engine.prefill(tf.constant([[1, 2]], tf.int32))
        model = engine.model
        with self.assertRaises(ValueError):
            model.decode_step(tf.constant([[3]]), caches[:1])
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model.decode_step(tf.constant([[3, 4]]), caches)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model.decode_step(tf.constant([[3], [4]]), caches)
        mismatch = ((caches[0][0], caches[0][1][:, :, :1]), caches[1])
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model.decode_step(tf.constant([[3]]), mismatch)
        _, full = engine.prefill(tf.ones([1, 8], tf.int32))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model.decode_step(tf.constant([[3]]), full)
        for invalid in (tf.zeros([1, 0], tf.int32), tf.zeros([1, 9], tf.int32)):
            with self.assertRaises(tf.errors.InvalidArgumentError):
                model.prefill(invalid)

    def test_sampling_matches_reference_across_context_limit(self):
        engine = self.make_engine()
        vocabulary = list("abcdefghijkl")
        cached = generate_text(engine, vocabulary, "abc", count=24)
        reference = generate_text(engine, vocabulary, "abc", count=24, cached=False)
        self.assertEqual(cached, reference)
        self.assertEqual(generate_text(engine, vocabulary, "abc", count=0), "abc")
        for arguments in ({"temperature": 0}, {"temperature": float("nan")}, {"count": -1}):
            with self.assertRaises(ValueError):
                generate_text(engine, vocabulary, "abc", **arguments)


if __name__ == "__main__":
    unittest.main()
