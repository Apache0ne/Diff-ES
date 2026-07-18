#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diff-ES SDXL entrypoint for complete single-file checkpoints.

This wrapper keeps the original pruning/search implementation unchanged while
replacing its hardcoded SDXL base pipeline load with
StableDiffusionXLPipeline.from_single_file(). The checkpoint's own UNet, two
text encoders, and VAE are loaded first; Diff-ES then wraps that live UNet with
UNet2DConditionPruned and performs calibration/search on those weights.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from diffusers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline as DiffusersStableDiffusionXLPipeline,
)

import evo_pruning_sdxl as diff_es


DEFAULT_MODEL_PATH = "/content/models/hyper_sdxl_4step_471056.safetensors"


def _set_offline_mode(local_files_only: bool) -> None:
    """Undo the original script's forced offline mode when downloads are allowed."""
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    # These libraries cache the environment state during import, so update the
    # cached flags as well. This is needed because evo_pruning_sdxl sets both
    # variables before this wrapper receives CLI arguments.
    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.HF_HUB_OFFLINE = bool(local_files_only)
    except Exception:
        pass

    try:
        import transformers.utils.hub as transformers_hub

        if hasattr(transformers_hub, "_is_offline_mode"):
            transformers_hub._is_offline_mode = bool(local_files_only)
    except Exception:
        pass


def _resolve_local_path(value: Optional[str], *, required: bool = False) -> Optional[str]:
    if value is None:
        return None

    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)

    if required:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Single-file SDXL checkpoint not found: {path}\n"
                "Pass the downloaded .safetensors or .ckpt file with --model-path."
            )
        if path.suffix.lower() not in {".safetensors", ".ckpt"}:
            raise ValueError(
                f"Unsupported checkpoint extension '{path.suffix}'. "
                "Expected .safetensors or .ckpt."
            )
        return str(path)

    # Config values may be either local paths or Hugging Face repository IDs.
    if path.exists():
        return str(path.resolve())
    return value


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model dtype: {name}") from exc


def _configure_scheduler(pipe: DiffusersStableDiffusionXLPipeline, name: str) -> None:
    if name == "checkpoint":
        return

    if name == "ddim-trailing":
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        return

    if name == "euler-trailing":
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        return

    raise ValueError(f"Unknown scheduler preset: {name}")


def _build_single_file_pipeline(args, inherited_kwargs):
    checkpoint_path = _resolve_local_path(args.model_path, required=True)
    config_source = _resolve_local_path(args.model_config)
    original_config_file = _resolve_local_path(args.original_config_file)
    dtype = _torch_dtype(args.model_dtype)

    _set_offline_mode(args.local_files_only)

    load_kwargs = dict(inherited_kwargs)
    # These are valid for from_pretrained() but not needed for a local
    # single-file conversion.
    load_kwargs.pop("variant", None)
    load_kwargs.pop("use_safetensors", None)

    load_kwargs["torch_dtype"] = dtype
    load_kwargs["local_files_only"] = bool(args.local_files_only)

    if config_source:
        load_kwargs["config"] = config_source
    if original_config_file:
        load_kwargs["original_config_file"] = original_config_file
    if args.disable_mmap:
        load_kwargs["disable_mmap"] = True

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        load_kwargs["token"] = hf_token

    logging.info("[Model] Loading complete SDXL checkpoint: %s", checkpoint_path)
    logging.info("[Model] Checkpoint dtype: %s", args.model_dtype)
    if config_source:
        logging.info("[Model] Pipeline config source: %s", config_source)
    else:
        logging.info("[Model] Pipeline config source: inferred by Diffusers")
    logging.info(
        "[Model] The UNet, text_encoder, text_encoder_2, and VAE weights come "
        "from the single checkpoint."
    )

    pipe = DiffusersStableDiffusionXLPipeline.from_single_file(
        checkpoint_path,
        **load_kwargs,
    )

    required_components = ("unet", "text_encoder", "text_encoder_2", "vae")
    missing_components = [
        name for name in required_components if getattr(pipe, name, None) is None
    ]
    if missing_components:
        raise RuntimeError(
            "The checkpoint did not produce a complete SDXL pipeline. Missing: "
            + ", ".join(missing_components)
        )

    _configure_scheduler(pipe, args.scheduler)
    logging.info("[Model] Scheduler preset: %s", args.scheduler)
    return pipe


def _count_parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _run_load_only(args) -> None:
    """Load and convert the checkpoint without requiring COCO or running search."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = _build_single_file_pipeline(args, {})
    pruned_unet = diff_es.attach_pruned_unet(pipe, device)
    pipe.to(device)

    n_basic, n_wrappers = diff_es.count_pruned_blocks(pruned_unet)
    components = {
        "unet": pipe.unet,
        "text_encoder": pipe.text_encoder,
        "text_encoder_2": pipe.text_encoder_2,
        "vae": pipe.vae,
    }

    print("=" * 80)
    print("DIFF-ES SINGLE-FILE CHECKPOINT LOAD TEST")
    print("=" * 80)
    print(f"Checkpoint: {Path(args.model_path).expanduser()}")
    print(f"Device:     {device}")
    print(f"Scheduler:  {pipe.scheduler.__class__.__name__}")
    for name, component in components.items():
        try:
            dtype = next(component.parameters()).dtype
        except StopIteration:
            dtype = "no parameters"
        print(
            f"{name:15s} params={_count_parameters(component):,} "
            f"dtype={dtype}"
        )
    print(f"Pruned BasicTransformerBlock modules: {n_basic}")
    print(f"Pruned Transformer2DModel wrappers:    {n_wrappers}")
    print("The complete checkpoint loaded and its UNet was converted for Diff-ES.")
    print("=" * 80)


def _install_single_file_loader(args) -> None:
    class SingleFilePipelineProxy:
        @classmethod
        def from_pretrained(cls, _ignored_model_id, **kwargs):
            # evo_pruning_sdxl.main() calls from_pretrained() once. Redirect that
            # call to the complete single-file checkpoint instead of loading
            # stabilityai/stable-diffusion-xl-base-1.0 weights.
            return _build_single_file_pipeline(args, kwargs)

    diff_es.StableDiffusionXLPipeline = SingleFilePipelineProxy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = diff_es.build_arg_parser()
    model_group = parser.add_argument_group("single-file SDXL checkpoint")

    model_group.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Complete SDXL .safetensors/.ckpt file containing the UNet, both "
            "text encoders, and VAE."
        ),
    )
    model_group.add_argument(
        "--model-config",
        type=str,
        default=None,
        help=(
            "Optional Diffusers config source: a local pipeline directory or "
            "a Hugging Face repo ID. No model weights are loaded from this "
            "source. When omitted, Diffusers infers the SDXL config."
        ),
    )
    model_group.add_argument(
        "--original-config-file",
        type=str,
        default=None,
        help="Optional original SDXL YAML config for unusual checkpoints.",
    )
    model_group.add_argument(
        "--model-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Dtype used while converting and loading the complete checkpoint.",
    )
    model_group.add_argument(
        "--scheduler",
        choices=["ddim-trailing", "euler-trailing", "checkpoint"],
        default="ddim-trailing",
        help=(
            "Scheduler preset. Hyper-SDXL four-step checkpoints normally use "
            "DDIM with trailing timestep spacing."
        ),
    )
    model_group.add_argument(
        "--load-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load the complete checkpoint, install the prunable UNet wrapper, "
            "print component statistics, and exit without COCO/search."
        ),
    )
    model_group.add_argument(
        "--disable-mmap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable safetensors mmap for files stored on slow/network drives.",
    )

    # The fork's target checkpoint is a Hyper-SDXL four-step model.
    parser.set_defaults(
        num_sampling_steps=4,
        cfg_scale=0.0,
        local_files_only=False,
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.load_only:
        _run_load_only(args)
        return
    _install_single_file_loader(args)
    diff_es.main(args)


if __name__ == "__main__":
    main()
