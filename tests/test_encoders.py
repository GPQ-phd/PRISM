import unittest

import torch

from models.prism import HFE, HFBlock, LFE
from models.vrwkv import VRWKVBlock


class HFETests(unittest.TestCase):
    def test_hfblock_has_five_depthwise_branches_and_grouped_branch_mixer(self):
        channels = 8
        block = HFBlock(channels)
        self.assertEqual(
            [(m.kernel_size, m.dilation) for m in block.branches],
            [((3, 3), (1, 1)), ((5, 5), (1, 1)), ((7, 7), (1, 1)), ((3, 3), (2, 2)), ((3, 3), (3, 3))],
        )
        self.assertTrue(all(m.groups == 3 * channels for m in block.branches))
        self.assertEqual(block.mix.groups, 3 * channels)
        self.assertEqual(block.mix.in_channels, 15 * channels)
        self.assertEqual(block.mix.out_channels, 3 * channels)

    def test_hfe_returns_target_feature_and_probability_prior(self):
        module = HFE(8, repeats=3)
        self.assertEqual(len(module.blocks), 3)
        bands = tuple(torch.randn(2, 3, 6, 5) for _ in range(3))
        energy = torch.cat([b.abs().mean((2, 3)) for b in bands], dim=1)
        feature, prior = module(bands, energy)
        self.assertEqual(feature.shape, (2, 8, 6, 5))
        self.assertEqual(prior.shape, (2, 1, 6, 5))
        self.assertTrue(((prior >= 0) & (prior <= 1)).all())


class LFETests(unittest.TestCase):
    def test_lfe_contains_two_vision_rwkv_blocks(self):
        module = LFE(8, blocks=2)
        self.assertEqual(len(module.blocks), 2)
        self.assertTrue(all(isinstance(block, VRWKVBlock) for block in module.blocks))

    def test_lfe_preserves_spatial_shape_and_has_bidirectional_influence(self):
        torch.manual_seed(3)
        module = LFE(8, blocks=2)
        low = torch.randn(1, 3, 2, 3, requires_grad=True)
        state = torch.randn(1, 44)
        output = module(low, state)
        self.assertEqual(output.shape, (1, 8, 2, 3))
        output[:, :, 0, 0].sum().backward()
        self.assertGreater(float(low.grad[:, :, -1, -1].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
