"""Synthetic regression for nonshared LLMDet checkpoint head restoration."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from llmdet_heads import restore_heads
from safetensors.torch import save_file
from torch import nn


def temporary_checkpoint():
    root = Path(__file__).resolve().parent / "local"
    root.mkdir(exist_ok=True)
    return TemporaryDirectory(prefix="llmdet-head-test-", dir=root)


class BoxHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 4)])


class ClassHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))


class BrokenModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="mm-grounding-dino",
            decoder_layers=6,
            decoder_bbox_embed_share=False,
            decoder_cls_embed_share=False,
        )
        self.bbox_embed = nn.ModuleList([BoxHead()] * 6)
        self.class_embed = nn.ModuleList([ClassHead()] * 6)
        self.model = nn.Module()
        self.model.decoder = nn.Module()
        self.model.decoder.bbox_embed = self.bbox_embed
        self.model.decoder.class_embed = self.class_embed
        self.to(dtype=dtype)


def checkpoint(model):
    tensors = {}
    for family in ("bbox_embed", "class_embed"):
        for layer, head in enumerate(getattr(model, family)):
            prefix = family if layer == 0 else f"model.decoder.{family}"
            for index, (suffix, value) in enumerate(head.state_dict().items()):
                tensors[f"{prefix}.{layer}.{suffix}"] = torch.full(
                    value.shape, layer + 0.125 * (index + 1), dtype=torch.float32
                )
    return tensors


class RestoreHeadsTests(unittest.TestCase):
    def test_all_layers_exact_and_independent_in_float32_and_float16(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype), temporary_checkpoint() as directory:
                model = BrokenModel(dtype)
                tensors = checkpoint(model)
                save_file(tensors, str(Path(directory) / "model.safetensors"))
                audit = restore_heads(model, directory)
                self.assertEqual(audit["restored_tensor_count"], 42)
                self.assertEqual(audit["restored_head_count"], 12)
                self.assertEqual(audit["max_absolute_error_after_dtype_device_conversion"], 0)
                for family in ("bbox_embed", "class_embed"):
                    heads = getattr(model, family)
                    self.assertIs(heads, getattr(model.model.decoder, family))
                    self.assertEqual(len({id(head) for head in heads}), 6)
                    for layer, head in enumerate(heads):
                        prefix = family if layer == 0 else f"model.decoder.{family}"
                        for suffix, actual in head.state_dict().items():
                            self.assertTrue(
                                torch.equal(actual, tensors[f"{prefix}.{layer}.{suffix}"].to(dtype))
                            )
                untouched = model.class_embed[1].bias.detach().clone()
                with torch.no_grad():
                    model.class_embed[0].bias.add_(7)
                self.assertTrue(torch.equal(model.class_embed[1].bias, untouched))

    def test_missing_late_class_bias_fails_without_any_mutation(self):
        with temporary_checkpoint() as directory:
            model = BrokenModel()
            original_bbox, original_class = model.bbox_embed, model.class_embed
            before = {key: value.clone() for key, value in model.state_dict().items()}
            tensors = checkpoint(model)
            del tensors["model.decoder.class_embed.5.bias"]
            save_file(tensors, str(Path(directory) / "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "Missing checkpoint head tensor"):
                restore_heads(model, directory)
            self.assertIs(model.bbox_embed, original_bbox)
            self.assertIs(model.class_embed, original_class)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))

    def test_shared_config_and_wrong_model_rejected(self):
        model = BrokenModel()
        model.config.decoder_bbox_embed_share = True
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            restore_heads(model, ".")
        model.config.model_type = "grounding-dino"
        with self.assertRaisesRegex(ValueError, "scoped to MM"):
            restore_heads(model, ".")

    def test_conflicting_checkpoint_alias_rejected(self):
        with temporary_checkpoint() as directory:
            model = BrokenModel()
            tensors = checkpoint(model)
            tensors["model.decoder.class_embed.0.bias"] = torch.tensor(999.0)
            save_file(tensors, str(Path(directory) / "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "Conflicting checkpoint aliases"):
                restore_heads(model, directory)


if __name__ == "__main__":
    unittest.main()
