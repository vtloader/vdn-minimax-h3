"""Repository-relative paths and the Hub fallback for weights.

A ModelSpec's `base.source`, a config's `checkpoint` / `base_source` / `vae_source` and
the render default are all written relative to the repository root
(`ckpts/<name>`), so an artifact carries no machine-specific path. When such a
directory is not present locally it is fetched from the Hub (`HF_WEIGHTS_REPO`, the
same layout) into the huggingface cache and used from there. Absolute paths pass through.
"""
import json
import os
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HF_WEIGHTS_REPO = "OpenVDN/vdn-minimax-h3"   # h3-base/ stage-b-step-2000/ stage-dmd-step-250/
WEIGHTS_PREFIX = "ckpts/"
LEGACY_WEIGHTS_PREFIX = "ckpts/converted/"   # how the training repo spells the same directories

H3_BASE = WEIGHTS_PREFIX + "h3-base"     # transformer/ vae/ audio_vae/ of the release
H3_UPSTREAM_REPO = "MiniMaxAI/MiniMax-H3"   # text_encoder/ tokenizer/ -- prompt encoding only
TEXT_ENCODER_ROOT = WEIGHTS_PREFIX + "text-encoder"


def resolve_repo_path(path):
    """Absolute path for `path`: unchanged if absolute or None, else joined to the
    repository root (not the cwd)."""
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def resolve_weights(path):
    """`resolve_repo_path`, plus the Hub fallback: a `ckpts/<name>` that does not exist
    locally is downloaded from HF_WEIGHTS_REPO (that subfolder only) and the cached copy
    is returned. `ckpts/converted/<name>` (a checkpoint exported by the training repo
    names its base that way) means the same directory."""
    if path is not None and path.startswith(LEGACY_WEIGHTS_PREFIX):
        path = WEIGHTS_PREFIX + path[len(LEGACY_WEIGHTS_PREFIX):]

    local = resolve_repo_path(path)
    if local is None or os.path.exists(local):
        return local

    if path.startswith(WEIGHTS_PREFIX):
        name = path[len(WEIGHTS_PREFIX):].strip("/")
        from huggingface_hub import snapshot_download

        root = snapshot_download(HF_WEIGHTS_REPO, allow_patterns=[f"{name}/**"])
        return os.path.join(root, name)

    return local


def upstream_snapshot(*subfolders):
    """Download only the conditioner into the project's weights directory.

    The text encoder alone is about 66.7 GB. Store it on the weights volume rather
    than in ~/.cache, and reuse a complete local copy without requiring the Hub.
    HTTP is the default transport: some Hub mirrors serve metadata but cannot obtain
    usable Xet/CAS credentials. HF_HUB_DISABLE_XET=0 opts back into Xet.
    """
    root = Path(resolve_repo_path(TEXT_ENCODER_ROOT))
    if _conditioner_complete(root, subfolders):
        print(f"Using local text encoder: {root}", flush=True)
        return str(root)

    from huggingface_hub import constants, snapshot_download

    # Transformers may already have imported huggingface_hub before this helper runs;
    # its constants are initialized at import time, so setting only os.environ here
    # would not change the transport used by file_download.
    if os.environ.get("HF_HUB_DISABLE_XET", "1").upper() in {"1", "ON", "YES", "TRUE"}:
        constants.HF_HUB_DISABLE_XET = True
    print(f"Downloading processor/text_encoder from {H3_UPSTREAM_REPO} into {root} "
          f"({'HTTP' if constants.HF_HUB_DISABLE_XET else 'Xet'}; resumable)", flush=True)
    return snapshot_download(
        H3_UPSTREAM_REPO,
        allow_patterns=[f"{s}/**" for s in subfolders],
        local_dir=str(root),
        max_workers=2,
    )


def _conditioner_complete(root, subfolders):
    """Check the required configs and every shard referenced by the weight index."""
    for folder in subfolders:
        directory = root / folder
        if folder == "processor":
            required = [directory / "preprocessor_config.json",
                        directory / "tokenizer_config.json", directory / "tokenizer.json"]
        elif folder == "text_encoder":
            index = directory / "model.safetensors.index.json"
            try:
                weights = json.loads(index.read_text())["weight_map"]
                shards = set(weights.values())
                if not shards or any(Path(name).name != name for name in shards):
                    return False
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                return False
            required = [directory / "config.json", index, *(directory / name for name in shards)]
        else:
            return False
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            return False
    return bool(subfolders)
