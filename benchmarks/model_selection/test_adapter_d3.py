"""CPU-only checks of the standalone d3 adapter's behavior-bearing adaptations."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import torch
from adapter_d3 import Adapter, _format_module, _linear_patch_forward, _noncausal_hidden
from PIL import Image


class D3AdapterTests(unittest.TestCase):
    def test_linear_patch_projection_matches_original_conv3d(self):
        torch.manual_seed(19)
        projection = torch.nn.Conv3d(3, 8, (2, 4, 4), stride=(2, 4, 4), bias=True)
        module = types.SimpleNamespace(proj=projection, embed_dim=8)
        flat = torch.randn(7, 3 * 2 * 4 * 4)
        expected = projection(flat.reshape(-1, 3, 2, 4, 4)).reshape(-1, 8)
        actual = _linear_patch_forward(module, flat)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_full_attention_reads_all_present_keys_and_scan_keeps_padding(self):
        calls = []
        scan_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])

        def layer(hidden, **kwargs):
            calls.append(kwargs)
            return hidden + 1

        def recurrent(**kwargs):
            self.assertIs(kwargs["attention_mask"], scan_mask)
            self.assertIsNone(kwargs["past_key_values"])
            return scan_mask

        model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                num_hidden_layers=2,
                layer_types=["linear_attention", "full_attention"],
            ),
            layers=[layer, layer],
            rotary_emb=lambda embeds, positions: ("rotary", positions),
            norm=lambda hidden: hidden,
        )
        embeds = torch.zeros(2, 3, 4)
        positions = torch.zeros(3, 2, 3, dtype=torch.long)
        fake_masking = types.SimpleNamespace(create_recurrent_attention_mask=recurrent)
        with patch.dict("sys.modules", {"transformers.masking_utils": fake_masking}):
            result = _noncausal_hidden(model, embeds, scan_mask, positions)
        self.assertIs(calls[0]["attention_mask"], scan_mask)
        # A key-only [B,1,1,L] mask is deliberately noncausal: every query
        # sees all real keys, including keys after the query position.
        torch.testing.assert_close(calls[1]["attention_mask"], scan_mask.bool()[:, None, None, :])
        self.assertEqual(tuple(calls[1]["attention_mask"].shape), (2, 1, 1, 3))
        self.assertFalse(calls[1]["use_cache"])
        torch.testing.assert_close(result, torch.full_like(embeds, 2))

    def test_prepare_preserves_native_tiles_and_binary_prompt_contract(self):
        adapter = Adapter.__new__(Adapter)
        adapter.format = _format_module()
        adapter.codes = ["A", "B"]
        adapter.max_length = None
        adapter.model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                text_config=types.SimpleNamespace(max_position_embeddings=1000)
            )
        )
        captured = []

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                captured.append((messages, kwargs))
                return messages[-1]["content"][-1]["text"]

            def __call__(self, **kwargs):
                self.last = kwargs
                return {
                    "input_ids": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                    "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
                    "mm_token_type_ids": torch.zeros(2, 3, dtype=torch.long),
                    "pixel_values": torch.zeros(2, 3),
                    "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
                }

        adapter.processor = Processor()
        tiles = [Image.new("RGB", (256, 256)), Image.new("RGB", (256, 256))]
        prepared = adapter.prepare(tiles, ["a swimming pool", "parked cars"])
        self.assertIs(adapter.processor.last["images"], tiles)
        self.assertEqual(tuple(prepared["input_ids"].shape), (2, 3))
        prompt = captured[0][0][-1]["content"][-1]["text"]
        self.assertEqual(
            prompt,
            (
                "State:\nAn aerial image tile.\n\nQuestion:\n"
                "Does this aerial image contain a swimming pool?\n\nOptions:\n"
                "A: No / false\nB: Yes / true\n\nReply with only the code of the best option."
            ),
        )
        self.assertTrue(captured[0][1]["add_generation_prompt"])
        self.assertFalse(captured[0][1]["enable_thinking"])
        with self.assertRaises(ValueError):
            adapter.prepare(tiles, ["only one query"])

    def test_postprocess_returns_true_probability_and_rejects_invalid_scores(self):
        adapter = Adapter.__new__(Adapter)
        prepared = {"input_ids": torch.ones(2, 3, dtype=torch.long)}
        rows = adapter.postprocess(torch.tensor([[0.9, 0.1], [0.2, 0.8]]), prepared)
        self.assertAlmostEqual(rows[0]["probability"], 0.1)
        self.assertAlmostEqual(rows[1]["probability"], 0.8)
        with self.assertRaises(ValueError):
            adapter.postprocess(torch.tensor([[float("nan"), 0.2], [0.1, 0.9]]), prepared)
        with self.assertRaises(ValueError):
            adapter.postprocess(torch.zeros(1, 2), prepared)


if __name__ == "__main__":
    unittest.main()
