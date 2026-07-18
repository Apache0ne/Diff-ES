#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report one SDXL transformer block skipped on the final three of four steps.

This wrapper reuses static_shared_block_ablation_report.py, but changes the
pruning policy from global/static removal to a stage-aware schedule:

- first high-noise denoising step (stage 3, t=750..999): full UNet
- second step (stage 2, t=500..749): skip the selected block
- third step  (stage 1, t=250..499): skip the selected block
- final step  (stage 0, t=0..249):   skip the selected block

The output remains one self-contained HTML file with metrics, dense/pruned
images, amplified difference images, configuration, and embedded JSON.
"""

from __future__ import annotations

import html

import static_shared_block_ablation_report as base
from evo_pruning_utils_sdxl import apply_layerdrop_schedule


STAGES = [
    (0, 249),
    (250, 499),
    (500, 749),
    (750, 999),
]


def set_last3_blockdrop(unet, block_id: int) -> None:
    """Keep the first/high-noise step dense; skip block on the last three."""
    base.clear_acceleration(unet)
    schedule = {
        0: [int(block_id)],
        1: [int(block_id)],
        2: [int(block_id)],
        3: [],
    }
    apply_layerdrop_schedule(unet, schedule, stages=STAGES)


_ORIGINAL_BUILD_HTML = base.build_html


def build_last3_html(*, args, rows, scheduler, device, component_counts) -> str:
    report = _ORIGINAL_BUILD_HTML(
        args=args,
        rows=rows,
        scheduler=scheduler,
        device=device,
        component_counts=component_counts,
    )

    block = int(args.block_id)
    steps = int(args.steps)
    final_steps = max(0, steps - 1)

    replacements = {
        "SDXL static shared-block ablation": "SDXL final-three-step block ablation",
        "SDXL shared block": "SDXL final-three-step block",
        (
            f"This is a screening result for block <b>{block}</b> skipped during "
            f"every one of the {steps} denoising steps. The threshold is intentionally "
            "strict because the goal is physical, universal removal rather than "
            "prompt-specific dynamic skipping."
        ): (
            f"This is a screening result for block <b>{block}</b>. The first high-noise "
            f"denoising step uses the full UNet; block {block} is skipped only during "
            f"the final {final_steps} denoising steps. The threshold remains strict "
            "because the goal is a stable scheduler-aware acceleration policy."
        ),
        f"Block {block} skipped at every step": f"Block {block} skipped on final {final_steps} steps",
        "removed_at_every_denoising_step&quot;: true": "removed_at_every_denoising_step&quot;: false",
    }
    for old, new in replacements.items():
        report = report.replace(old, new)

    schedule_note = (
        "<br><b>Block-drop policy:</b> first/high-noise step = full UNet; "
        f"final {final_steps} steps = block {block} skipped"
        "<br><b>Stage schedule:</b> stage 3: []; stages 2, 1, 0: ["
        f"{block}]"
    )
    report = report.replace(
        f"<b>Steps:</b> {steps}",
        f"<b>Steps:</b> {steps}{schedule_note}",
        1,
    )

    embedded_extra = html.escape(
        '\n"drop_policy": "first high-noise step dense; selected block skipped on final three steps",'
        f'\n"stage_schedule": {{"0": [{block}], "1": [{block}], "2": [{block}], "3": []}},'
    )
    report = report.replace(
        "<section class=\"card\"><h2>Embedded machine-readable report</h2><pre>{",
        "<section class=\"card\"><h2>Embedded machine-readable report</h2><pre>{" + embedded_extra,
        1,
    )
    return report


# Existing generate() resolves these functions from the base module at runtime.
base.set_static_blockdrop = set_last3_blockdrop
base.build_html = build_last3_html


if __name__ == "__main__":
    base.main()
