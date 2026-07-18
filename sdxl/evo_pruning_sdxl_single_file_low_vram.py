#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Low-VRAM entrypoint for complete single-file SDXL Diff-ES runs.

This reuses evo_pruning_sdxl_single_file.py, but configures the VAE before the
original Diff-ES main function moves the pipeline to CUDA:

- keep the SDXL VAE permanently in float32 instead of repeatedly upcasting and
  downcasting it for every calibration/search decode;
- cast incoming latent tensors to the live VAE dtype before every decode;
- enable VAE batch slicing so a fitness probe batch is decoded one image at a
  time;
- enable VAE tiling so 1024x1024 decoding has a lower activation-memory peak.

The pruning/search arguments and experiment naming remain unchanged, so a run
that previously completed LayerDrop calibration can reuse its cached order
files when restarted with this entrypoint.
"""

import logging

import torch

import evo_pruning_sdxl_single_file as single_file


_ORIGINAL_BUILD_PIPELINE = single_file._build_single_file_pipeline


def _install_vae_dtype_guard(vae) -> None:
    """Make direct pipeline VAE decodes match the VAE parameter dtype.

    Diffusers normally casts latents while temporarily upcasting an FP16 VAE.
    When the VAE is already permanently FP32, that temporary-upcast branch is
    skipped, so the pipeline can pass FP16 latents into FP32 convolutions. This
    instance-level wrapper performs the missing cast without modifying
    Diffusers itself.
    """
    original_decode = vae.decode

    def decode_with_matching_dtype(z, *args, **kwargs):
        try:
            vae_dtype = next(vae.parameters()).dtype
        except StopIteration:
            vae_dtype = z.dtype

        if z.dtype != vae_dtype:
            z = z.to(dtype=vae_dtype)
        return original_decode(z, *args, **kwargs)

    vae.decode = decode_with_matching_dtype


def _build_low_vram_pipeline(args, inherited_kwargs):
    pipe = _ORIGINAL_BUILD_PIPELINE(args, inherited_kwargs)

    # Keep the VAE in FP32. The decode guard below ensures FP16 UNet latents are
    # converted before they reach the VAE's FP32 post_quant_conv.
    pipe.vae.to(dtype=torch.float32)
    _install_vae_dtype_guard(pipe.vae)

    # Use the component-level APIs to avoid the deprecated pipeline wrappers.
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    logging.info(
        "[Memory] VAE kept in float32; decode dtype guard, slicing, and tiling enabled."
    )
    return pipe


# The base wrapper resolves this global when loading the model, including from
# its SingleFilePipelineProxy, so replacing it here affects both --load-only
# and full pruning/search runs without duplicating the original implementation.
single_file._build_single_file_pipeline = _build_low_vram_pipeline


if __name__ == "__main__":
    single_file.main()
