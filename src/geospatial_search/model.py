"""Shared, revision-pinned GeoRSCLIP loading and normalization contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .download_nys_naip import sha256
from .paths import CONFIG


def load_config(path: Path = CONFIG) -> dict:
    config = json.loads(path.read_text(encoding='utf-8'))
    required = {'dataset_id','chip_size','stride','min_valid_fraction','model_id','model_filename',
                'model_revision','arch','base_pretrained','batch_size','workers','prefetch_batches','amp'}
    if set(config) != required:
        raise ValueError(f'Configuration fields must be {sorted(required)}')
    if len(config['model_revision']) != 40:
        raise ValueError('Pin the model to a full Hugging Face commit revision')
    for name in ('batch_size','workers','prefetch_batches'):
        if not isinstance(config[name],int) or config[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    return config


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def normalized(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features,dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('Model features must be a finite matrix')
    norms = np.linalg.norm(features,axis=1,keepdims=True)
    if np.any(norms <= 0):
        raise ValueError('Model produced a zero-length embedding')
    return features/norms


def load_model(config: dict, *, cache_dir: Path | None = None, offline: bool = False):
    import open_clip
    import torch
    from huggingface_hub import hf_hub_download

    if not torch.cuda.is_available():
        raise RuntimeError('No GPU available. Install the appropriate ROCm or CUDA PyTorch build; see the GPU setup guide.')
    device = torch.device('cuda:0')
    checkpoint = Path(hf_hub_download(repo_id=config['model_id'],filename=config['model_filename'],
                                   revision=config['model_revision'],cache_dir=str(cache_dir) if cache_dir else None,
                                   local_files_only=offline))
    # Build the same architecture/preprocessing without downloading redundant base weights.
    pretrained_cfg = open_clip.get_pretrained_cfg(config['arch'],config['base_pretrained'])
    if not pretrained_cfg:
        raise ValueError('Unknown base preprocessing configuration')
    model,_,preprocess = open_clip.create_model_and_transforms(
        config['arch'],pretrained=None,
        image_mean=pretrained_cfg.get('mean'),image_std=pretrained_cfg.get('std'),
        image_interpolation=pretrained_cfg.get('interpolation','bicubic'),
        image_resize_mode=pretrained_cfg.get('resize_mode','shortest'))
    checkpoint_data = torch.load(checkpoint,map_location='cpu',weights_only=True)
    state = checkpoint_data.get('state_dict',checkpoint_data) if isinstance(checkpoint_data,dict) else checkpoint_data
    state = {key.removeprefix('module.'):value for key,value in state.items()}
    model.load_state_dict(state,strict=True)
    del checkpoint_data,state
    contract = {'model_id':config['model_id'],'model_filename':config['model_filename'],
                'model_revision':config['model_revision'],'arch':config['arch'],
                'base_pretrained':config['base_pretrained'],'model_fingerprint':sha256(checkpoint),
                'preprocess_fingerprint':hashlib.sha256(repr(preprocess).encode()).hexdigest(),
                'preprocess':repr(preprocess),'model_cache_path':str(checkpoint),
                'embedding_dim':int(model.text_projection.shape[-1]),'embedding_dtype':'float16',
                'metric':'cosine (L2-normalized inner product)','rgb_band_order':[1,2,3],
                'torch_version':torch.__version__,'open_clip_version':open_clip.__version__,
                'gpu':torch.cuda.get_device_name(0),'amp':config['amp']}
    return model.to(device).eval(),preprocess,device,contract


def encode_images(model, inputs, device, amp: bool) -> np.ndarray:
    import torch
    # The installed Windows ROCm pinned allocator hangs at interpreter shutdown.
    # Ordinary CPU buffers are reliable; CUDA builds can use pinned transfers.
    if torch.cuda.is_available() and not torch.version.hip and not inputs.is_pinned():
        inputs = inputs.pin_memory()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):
        raw = model.encode_image(inputs.to(device,non_blocking=True))
    return normalized(raw.float().cpu().numpy())


def encode_text(model, text: str, arch: str, device, amp: bool) -> np.ndarray:
    import open_clip
    import torch
    tokens = open_clip.get_tokenizer(arch)([text]).to(device)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):
        raw = model.encode_text(tokens)
    return normalized(raw.float().cpu().numpy())[0]
