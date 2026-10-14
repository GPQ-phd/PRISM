import unittest

import torch

from models.vrwkv import BiWKV, ChannelMix, SpatialMix, q_shift


def direct_biwkv(k, v, decay, first):
    """Small reference implementation of Vision-RWKV Equation 5."""
    batch, tokens, channels = k.shape
    output = torch.empty_like(v)
    for t in range(tokens):
        exponents = []
        for i in range(tokens):
            if i == t:
                exponent = first + k[:, i]
            else:
                exponent = -((abs(t - i) - 1) / tokens) * decay + k[:, i]
            exponents.append(exponent)
        scores = torch.stack(exponents, dim=1)
        weights = torch.softmax(scores, dim=1)
        output[:, t] = (weights * v).sum(dim=1)
    return output


class QShiftTests(unittest.TestCase):
    def test_q_shift_moves_four_channel_groups(self):
        image = torch.arange(1, 1 + 4 * 3 * 3, dtype=torch.float32).reshape(1, 4, 3, 3)
        tokens = image.flatten(2).transpose(1, 2)
        shifted = q_shift(tokens, (3, 3), gamma=0.25).transpose(1, 2).reshape_as(image)

        expected = torch.zeros_like(image)
        expected[:, 0, :, 1:] = image[:, 0, :, :-1]
        expected[:, 1, :, :-1] = image[:, 1, :, 1:]
        expected[:, 2, 1:, :] = image[:, 2, :-1, :]
        expected[:, 3, :-1, :] = image[:, 3, 1:, :]
        self.assertTrue(torch.equal(shifted, expected))

    def test_q_shift_preserves_singleton_spatial_shapes(self):
        for resolution in ((1, 5), (5, 1), (1, 1)):
            x = torch.randn(2, resolution[0] * resolution[1], 8)
            self.assertEqual(q_shift(x, resolution).shape, x.shape)

    def test_q_shift_rejects_inconsistent_token_count(self):
        with self.assertRaises(ValueError):
            q_shift(torch.randn(1, 8, 4), (3, 3))


class BiWKVTests(unittest.TestCase):
    def test_biwkv_matches_direct_bidirectional_equation(self):
        torch.manual_seed(2)
        k = torch.randn(2, 5, 3, dtype=torch.float64)
        v = torch.randn(2, 5, 3, dtype=torch.float64)
        layer = BiWKV(3, chunk_size=2).double()
        with torch.no_grad():
            layer.spatial_decay.copy_(torch.tensor([-1.2, 0.4, 1.1], dtype=torch.float64))
            layer.spatial_first.copy_(torch.tensor([0.2, -0.3, 0.7], dtype=torch.float64))
        actual = layer(k, v)
        expected = direct_biwkv(k, v, layer.spatial_decay, layer.spatial_first)
        # BiWKV deliberately accumulates in float32 for AMP stability.
        self.assertTrue(torch.allclose(actual, expected, atol=3e-7, rtol=1e-6))

    def test_biwkv_uses_future_tokens(self):
        layer = BiWKV(1)
        with torch.no_grad():
            layer.spatial_decay.zero_()
            layer.spatial_first.zero_()
        k = torch.zeros(1, 4, 1)
        v = torch.tensor([[[0.0], [0.0], [0.0], [4.0]]])
        self.assertGreater(float(layer(k, v)[0, 0, 0]), 0.0)

    def test_biwkv_backward_reaches_keys_values_and_parameters(self):
        layer = BiWKV(2)
        k = torch.randn(1, 4, 2, requires_grad=True)
        v = torch.randn(1, 4, 2, requires_grad=True)
        layer(k, v).square().mean().backward()
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)
        self.assertIsNotNone(layer.spatial_decay.grad)
        self.assertIsNotNone(layer.spatial_first.grad)

    def test_biwkv_float32_accumulation_is_finite(self):
        layer = BiWKV(2)
        k = torch.tensor([[[80.0, -80.0], [-80.0, 80.0], [60.0, 60.0]]])
        v = torch.randn(1, 3, 2)
        self.assertTrue(torch.isfinite(layer(k, v)).all())


class MixerTests(unittest.TestCase):
    def test_spatial_mix_uses_q_shift_and_biwkv(self):
        module = SpatialMix(8, num_layers=2, layer_id=0)
        self.assertIsInstance(module.biwkv, BiWKV)
        y = module(torch.randn(1, 9, 8), (3, 3))
        self.assertEqual(y.shape, (1, 9, 8))

    def test_channel_mix_has_squared_relu_expansion(self):
        module = ChannelMix(8, num_layers=2, layer_id=0, hidden_rate=4)
        self.assertEqual(module.key.out_features, 32)
        self.assertEqual(module.value.in_features, 32)
        y = module(torch.randn(1, 9, 8), (3, 3))
        self.assertEqual(y.shape, (1, 9, 8))


if __name__ == "__main__":
    unittest.main()
