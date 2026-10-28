import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_transfer.networks import (
    DomainDiscriminator,
    GradientReversalLayer,
    ReconstructionNetwork,
    SharedSpecificRepresentation,
)
from feature_transfer.pretraining import LossFunctions, pretrain_model


class ArchitectureTests(unittest.TestCase):
    def test_representation_reconstruction_and_domain_shapes(self):
        model = SharedSpecificRepresentation(d_in=15, hidden_dims=[8, 4])
        reconstruction = ReconstructionNetwork(d_in=15, h_specific_dim=15, h_share_dim=15)
        discriminator = DomainDiscriminator(input_dim=15)
        source = torch.randn(4, 15)
        target = torch.randn(5, 15)

        source_specific, target_specific, source_shared, target_shared = model(source, target)
        self.assertEqual(source_specific.shape, source.shape)
        self.assertEqual(target_specific.shape, target.shape)
        self.assertEqual(source_shared.shape, source.shape)
        self.assertEqual(target_shared.shape, target.shape)
        self.assertEqual(reconstruction(source_shared, source_specific).shape, source.shape)
        self.assertEqual(discriminator(source_shared).shape, (4, 1))

    def test_source_pretraining_step_updates_parameters(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        model = SharedSpecificRepresentation(d_in=15, hidden_dims=[8, 4])
        reconstruction = ReconstructionNetwork(d_in=15, h_specific_dim=15, h_share_dim=15)
        discriminator = DomainDiscriminator(input_dim=15)
        before = next(model.parameters()).detach().clone()

        pretrain_model(
            torch.randn(4, 15),
            torch.randn(5, 15),
            model,
            reconstruction,
            discriminator,
            LossFunctions(beta=0.1, gamma=1e-6),
            epochs=1,
        )

        self.assertFalse(torch.equal(before, next(model.parameters()).detach()))

    def test_gradient_reversal_changes_only_input_gradient_sign(self):
        x = torch.tensor([2.0], requires_grad=True)
        weight = torch.tensor([3.0], requires_grad=True)
        (GradientReversalLayer.apply(x, 1.0) * weight).sum().backward()
        self.assertEqual(x.grad.item(), -3.0)
        self.assertEqual(weight.grad.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
