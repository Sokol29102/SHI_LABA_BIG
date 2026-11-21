import tempfile
import unittest
from pathlib import Path

import torch

from cnn import save_checkpoint
from vit import build_model, prepare_batch, load_model, ARCHITECTURE


class ViTChecks(unittest.TestCase):
    def test_preprocessing_resizes_and_preserves_eval_values(self):
        raw = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)
        raw[1] = 255
        original = raw.clone()
        x = prepare_batch(raw, [0.5]*3, [0.5]*3, augment=False)
        self.assertEqual(tuple(x.shape), (2, 3, 224, 224))
        torch.testing.assert_close(x[0], torch.full((3, 224, 224), -1.))
        torch.testing.assert_close(x[1], torch.ones(3, 224, 224))
        torch.testing.assert_close(raw, original)

    def test_pretrained_boundary_keeps_backbone_and_resets_only_head(self):
        torch.set_num_threads(2)
        reference = build_model()
        state = reference.state_dict()
        state['head.weight'] = torch.zeros(1000, 192)
        state['head.bias'] = torch.zeros(1000)
        state['cls_token'] = torch.full_like(state['cls_token'], 0.25)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'upstream.pt'
            torch.save(state, path)
            model = build_model(pretrained_path=path)
            self.assertEqual(model.head.out_features, 10)
            torch.testing.assert_close(model.cls_token, torch.full_like(model.cls_token, 0.25))
            self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_checkpoint_load_is_offline_and_preserves_predictions(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = build_model().eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.inference_mode():
            expected = model(x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'best.pt'
            save_checkpoint(path, {'architecture': ARCHITECTURE, 'model_state': model.state_dict(),
                'mean': [0.5]*3, 'std': [0.5]*3, 'classes': [str(i) for i in range(10)]})
            loaded, _ = load_model(path, 'cpu')
            with torch.inference_mode():
                torch.testing.assert_close(loaded(x), expected)


if __name__ == '__main__':
    unittest.main()
