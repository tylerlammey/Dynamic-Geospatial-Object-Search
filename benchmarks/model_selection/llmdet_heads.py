"""Restore checkpoint-specific MM Grounding DINO heads after the HF tying bug.

Scoped to the standalone benchmark. Transformers issue 49529 / PR 49552 describe
the unconditional layer ties that conflict with nonshared checkpoint heads.
Only head tensors are materialized, on CPU, from an existing local snapshot.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn


def restore_heads(model: nn.Module, snapshot_path: str | Path) -> dict:
    """Restore all nonshared bbox/class heads, failing before mutation on error.

    The root head ModuleLists are deliberately shared with the decoder's lists;
    individual layers within each list must have independent parameter storage.
    Missing or conflicting checkpoint tensors are errors, never random defaults.
    """
    config = model.config
    if getattr(config, "model_type", None) != "mm-grounding-dino":
        raise ValueError("Head repair is scoped to MM Grounding DINO checkpoints")
    for flag in ("decoder_bbox_embed_share", "decoder_cls_embed_share"):
        if getattr(config, flag, None) is not False:
            raise ValueError(f"Head repair requires explicit {flag}=False")
    layers = getattr(config, "decoder_layers", 0)
    if not isinstance(layers, int) or layers < 1:
        raise ValueError("Invalid decoder layer count")
    decoder = model.model.decoder
    for family in ("bbox_embed", "class_embed"):
        heads = getattr(model, family)
        if not isinstance(heads, nn.ModuleList) or len(heads) != layers:
            raise ValueError(f"Expected {layers} heads in {family}")
        if not hasattr(decoder, family):
            raise ValueError(f"Decoder has no {family}")

    snapshot = Path(snapshot_path).resolve()
    files = sorted(snapshot.glob("*.safetensors"))
    if not files:
        raise ValueError("No local safetensors checkpoint files found")
    restored = {}
    audit_rows = []
    max_error = 0.0
    with ExitStack() as stack:
        handles = [
            stack.enter_context(safe_open(str(path), framework="pt", device="cpu", backend="mmap"))
            for path in files
        ]
        locations = {}
        for path, handle in zip(files, handles, strict=True):
            for key in handle.keys():
                if key in locations:
                    raise ValueError(f"Duplicate checkpoint tensor key: {key}")
                locations[key] = (path, handle)

        for family in ("bbox_embed", "class_embed"):
            heads = []
            for layer in range(layers):
                # Copy each head separately; deepcopy of an already-aliased
                # ModuleList would preserve the erroneous layer aliasing.
                candidate = deepcopy(getattr(model, family)[layer])
                expected = {}
                for suffix, target in candidate.state_dict().items():
                    keys = [
                        f"{family}.{layer}.{suffix}",
                        f"model.decoder.{family}.{layer}.{suffix}",
                    ]
                    matches = [key for key in keys if key in locations]
                    if not matches:
                        raise ValueError(f"Missing checkpoint head tensor: {keys}")
                    key = matches[0]
                    path, handle = locations[key]
                    source = handle.get_tensor(key)
                    for duplicate in matches[1:]:
                        other = locations[duplicate][1].get_tensor(duplicate)
                        if source.dtype != other.dtype or not torch.equal(source, other):
                            raise ValueError(f"Conflicting checkpoint aliases: {matches}")
                    if source.shape != target.shape:
                        raise ValueError(
                            f"Wrong checkpoint shape for {key}: {source.shape} != {target.shape}"
                        )
                    expected[suffix] = source.to(device=target.device, dtype=target.dtype)
                    audit_rows.append({"file": path.name, "key": key, "shape": list(source.shape)})
                if not expected:
                    raise ValueError(f"No parameters found for {family}.{layer}")
                candidate.load_state_dict(expected, strict=True)
                for suffix, actual in candidate.state_dict().items():
                    reference = expected[suffix]
                    if not torch.equal(actual, reference):
                        raise ValueError(
                            f"Head tensor does not equal checkpoint after conversion: {family}.{layer}.{suffix}"
                        )
                    error = float(
                        (actual.detach().double() - reference.detach().double()).abs().max().cpu()
                    )
                    max_error = max(max_error, error)
                heads.append(candidate)
            restored[family] = nn.ModuleList(heads).train(getattr(model, family).training)

    # All checkpoint reads and strict loads succeeded before touching the model.
    for family, heads in restored.items():
        storages = set()
        for head in heads:
            for tensor in head.state_dict().values():
                if tensor.numel() == 0:
                    continue
                identity = (str(tensor.device), tensor.untyped_storage().data_ptr())
                if identity in storages:
                    raise ValueError(f"Head parameters still alias across {family} layers")
                storages.add(identity)
    for family, heads in restored.items():
        setattr(model, family, heads)
        setattr(decoder, family, heads)
    return {
        "applied": True,
        "reason": "restore_checkpoint_nonshared_heads_after_Transformers_49529",
        "decoder_layers": layers,
        "restored_head_count": 2 * layers,
        "restored_tensor_count": len(audit_rows),
        "max_absolute_error_after_dtype_device_conversion": max_error,
        "independent_layer_storage": True,
        "decoder_root_lists_identical": all(
            getattr(decoder, family) is getattr(model, family) for family in restored
        ),
        "checkpoint_directory": snapshot.name,
        "checkpoint_tensors": audit_rows,
    }
