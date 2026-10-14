import unittest

import torch
from torch.nn import functional as F

from models.prism import Metadata, SSAN


class MetadataTests(unittest.TestCase):
    def test_missing_metadata_uses_learned_default_embedding(self):
        module = Metadata()
        x = torch.randn(3, 3, 8, 8)
        embedding, mask = module(x)
        self.assertIsInstance(module.default_embedding, torch.nn.Parameter)
        self.assertTrue(torch.equal(mask, torch.zeros(3, dtype=torch.bool)))
        self.assertTrue(torch.allclose(embedding, module.default_embedding[None].expand(3, -1)))

    def test_mixed_batch_selects_encoded_and_default_rows(self):
        module = Metadata()
        x = torch.randn(3, 3, 8, 8)
        height = torch.tensor([10.0, float("nan"), 40.0])
        angle = torch.tensor([30.0, float("nan"), 60.0])
        requested = torch.tensor([True, False, True])
        embedding, mask = module(x, height, angle, requested)
        features = torch.stack(
            [height[[0, 2]].log(), torch.deg2rad(angle[[0, 2]]).sin(), torch.deg2rad(angle[[0, 2]]).cos()],
            dim=-1,
        )
        expected_valid = module.net(features)
        self.assertTrue(torch.equal(mask, requested))
        self.assertTrue(torch.allclose(embedding[[0, 2]], expected_valid))
        self.assertTrue(torch.allclose(embedding[1], module.default_embedding))

    def test_default_embedding_receives_gradient_for_missing_rows(self):
        module = Metadata()
        x = torch.randn(2, 3, 8, 8)
        embedding, _ = module(
            x,
            torch.tensor([12.0, float("nan")]),
            torch.tensor([25.0, float("nan")]),
            torch.tensor([True, False]),
        )
        embedding.sum().backward()
        self.assertIsNotNone(module.default_embedding.grad)
        self.assertTrue(torch.equal(module.default_embedding.grad, torch.ones(32)))
        self.assertIsNotNone(module.net[0].weight.grad)

    def test_invalid_metadata_batch_length_raises(self):
        module = Metadata()
        with self.assertRaises(ValueError):
            module(torch.randn(2, 3, 8, 8), [10.0, 20.0, 30.0], [1.0, 2.0, 3.0])


class SSANTests(unittest.TestCase):
    def test_alpha_initialization_and_token_shape(self):
        module = SSAN(6)
        self.assertAlmostEqual(float(module.alpha.detach()), 0.01)
        x = torch.randn(2, 5, 6)
        state = torch.randn(2, 44)
        self.assertEqual(module(x, state).shape, x.shape)

    def test_zero_alpha_reduces_to_layer_norm(self):
        module = SSAN(4)
        with torch.no_grad():
            module.alpha.zero_()
        x = torch.randn(2, 4, 3, 5)
        state = torch.randn(2, 44)
        actual = module(x, state)
        tokens = x.flatten(2).transpose(1, 2)
        expected = F.layer_norm(tokens, (4,)).transpose(1, 2).reshape_as(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
