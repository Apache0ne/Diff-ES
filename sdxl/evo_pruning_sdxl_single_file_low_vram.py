#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Low-VRAM entrypoint for complete single-file SDXL Diff-ES runs.

This reuses evo_pruning_sdxl_single_file.py, but configures the VAE before the
original Diff-ES main function moves the pipeline to CUDA:

- keep the SDXL VAE permanently in float32 instead of repeatedly upcasting and
  downcasting it for every calibration/search decode;
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


def _build_low_vram_pipeline(args, inherited_kwargs):
    pipe = _ORIGINAL_BUILD_PIPELINE(args, inherited_kwargs)

    # SDXL normally upcasts the VAE temporarily at every decode. Keeping it in
    # FP32 avoids repeated dtype conversions and allocator fragmentation.
    pipe.vae.to(dtype=torch.float32)

    # Slice batch decoding and tile spatial decoding to reduce peak VRAM.
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    logging.info("[Memory] VAE kept in float32 with slicing and tiling enabled.")
    return pipe


# The base wrapper resolves this global when loading the model, including from
# its SingleFilePipelineProxy, so replacing it here affects both --load-only
# and full pruning/search runs without duplicating the original implementation.
single_file._build_single_file_pipeline = _build_low_vram_pipeline


if __name__ == "__main__":
    single_file.main()
