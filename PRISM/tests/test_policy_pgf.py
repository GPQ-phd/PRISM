import unittest

import torch

from models.prism import PGF, Policy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.policy = Policy(4)
        self.low = torch.randn(2, 4, 3, 3)
        self.high = torch.randn(2, 4, 3, 3)
        self.state = torch.randn(2, 44)
        self.metadata = torch.randn(2, 32)
        self.mask = torch.tensor([False, True])

    def test_temperature_parameters_and_per_sample_formula(self):
        self.policy.eval()
        result = self.policy(self.low, self.high, self.state, self.metadata, self.mask)
        self.assertIsInstance(self.policy.tau0, torch.nn.Parameter)
        self.assertIsInstance(self.policy.delta_tau, torch.nn.Parameter)
        self.assertAlmostEqual(float(self.policy.tau0.detach()), 1.0)
        self.assertAlmostEqual(float(self.policy.delta_tau.detach()), 0.5)
        projected = (self.metadata * self.policy.temp_vector).sum(-1)
        expected_meta = self.policy.tau0 + self.policy.delta_tau * projected.sigmoid()
        self.assertTrue(torch.allclose(result["tau"][0], self.policy.tau0))
        self.assertTrue(torch.allclose(result["tau"][1], expected_meta[1]))

    def test_temperature_parameters_receive_gradients(self):
        result = self.policy(self.low, self.high, self.state, self.metadata, self.mask)
        result["tau"].sum().backward()
        self.assertIsNotNone(self.policy.tau0.grad)
        self.assertIsNotNone(self.policy.delta_tau.grad)

    def test_inference_is_deterministic_and_probabilities_are_normalized(self):
        self.policy.eval()
        first = self.policy(self.low, self.high, self.state, self.metadata, self.mask)
        second = self.policy(self.low, self.high, self.state, self.metadata, self.mask)
        self.assertTrue(torch.equal(first["probabilities"], second["probabilities"]))
        sums = first["probabilities"].flatten(1).sum(-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))


class PGFTests(unittest.TestCase):
    def test_causal_kernel_has_exact_local_forward_range(self):
        pgf = PGF(1, c=1)
        with torch.no_grad():
            pgf.causal.weight.fill_(1.0)
            pgf.causal.bias.zero_()
        sequence = torch.zeros(1, 12, 1)
        sequence[0, 3, 0] = 1.0
        refined = pgf.causal_refine(sequence, torch.ones_like(sequence))
        nonzero = refined[0, :, 0].nonzero().flatten().tolist()
        self.assertEqual(nonzero, list(range(3, 10)))
        self.assertEqual(pgf.causal.kernel_size, (7,))
        self.assertEqual(pgf.causal.groups, 1)

    def test_policy_head_receives_gradient_through_continuous_pgf_paths(self):
        torch.manual_seed(5)
        policy = Policy(4)
        policy.eval()
        low = torch.randn(1, 4, 3, 3)
        high = torch.randn(1, 4, 3, 3)
        state = torch.randn(1, 44)
        metadata = torch.randn(1, 32)
        result = policy(low, high, state, metadata, torch.tensor([True]))
        pgf = PGF(4, c=8)
        prior = torch.sigmoid(torch.randn(1, 1, 3, 3))
        pgf(low, high, prior, result["probabilities"], state).sum().backward()
        grad = policy.logits[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_mixing_head_is_depthwise_separable(self):
        pgf = PGF(4, c=8)
        self.assertEqual(pgf.mix[0].groups, 32)
        self.assertEqual(pgf.mix[1].kernel_size, (1, 1))


if __name__ == "__main__":
    unittest.main()
