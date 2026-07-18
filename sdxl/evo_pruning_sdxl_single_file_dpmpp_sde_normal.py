#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DPM++ SDE normal-schedule entrypoint for single-file SDXL Diff-ES.

This composes the existing single-file and low-VRAM wrappers, then adds the
Diffusers scheduler corresponding to the A1111/ComfyUI-style ``DPM++ SDE``
sampler with the ordinary (non-Karras) noise schedule.

Important distinction:
- DPM++ SDE      -> DPMSolverSinglestepScheduler
- DPM++ 2M SDE   -> DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++")

The preset below intentionally uses the singlestep SDE solver, second-order
midpoint updates, no Karras/exponential/beta sigma remapping, and a zero final
sigma. The low-VRAM module keeps the FP32 VAE decode guard, slicing, and tiling.
"""

import argparse
import logging

from diffusers import DPMSolverSinglestepScheduler

import evo_pruning_sdxl_single_file as single_file
# Import for its side effect: it replaces single_file._build_single_file_pipeline
# with the FP32-VAE low-memory loader and dtype guard.
import evo_pruning_sdxl_single_file_low_vram  # noqa: F401


_ORIGINAL_CONFIGURE_SCHEDULER = single_file._configure_scheduler
_ORIGINAL_BUILD_ARG_PARSER = single_file.build_arg_parser


def _configure_scheduler(pipe, name: str) -> None:
    if name != "dpmpp-sde-normal":
        _ORIGINAL_CONFIGURE_SCHEDULER(pipe, name)
        return

    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
        pipe.scheduler.config,
        solver_order=2,
        algorithm_type="sde-dpmsolver++",
        solver_type="midpoint",
        lower_order_final=True,
        thresholding=False,
        use_karras_sigmas=False,
        use_exponential_sigmas=False,
        use_beta_sigmas=False,
        final_sigmas_type="zero",
        steps_offset=0,
    )

    logging.info(
        "[Scheduler] DPM++ SDE normal: class=%s algorithm=%s order=%s "
        "solver=%s karras=%s exponential=%s beta=%s final_sigma=%s",
        pipe.scheduler.__class__.__name__,
        pipe.scheduler.config.algorithm_type,
        pipe.scheduler.config.solver_order,
        pipe.scheduler.config.solver_type,
        pipe.scheduler.config.use_karras_sigmas,
        pipe.scheduler.config.use_exponential_sigmas,
        pipe.scheduler.config.use_beta_sigmas,
        pipe.scheduler.config.final_sigmas_type,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = _ORIGINAL_BUILD_ARG_PARSER()
    for action in parser._actions:
        if action.dest == "scheduler":
            action.choices = [
                "dpmpp-sde-normal",
                "ddim-trailing",
                "euler-trailing",
                "checkpoint",
            ]
            action.default = "dpmpp-sde-normal"
            action.help = (
                "Scheduler preset. 'dpmpp-sde-normal' uses "
                "DPMSolverSinglestepScheduler with sde-dpmsolver++, order 2, "
                "midpoint updates, and ordinary non-Karras sigmas."
            )
            break
    else:
        raise RuntimeError("Inherited parser does not define --scheduler")
    return parser


single_file._configure_scheduler = _configure_scheduler
single_file.build_arg_parser = build_arg_parser


if __name__ == "__main__":
    single_file.main()
