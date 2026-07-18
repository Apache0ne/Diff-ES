#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Static all-step BasicTransformerBlock ablation report for single-file SDXL.

The script compares the original dense UNet against the same UNet with one
BasicTransformerBlock skipped for every denoising step. It uses DPM++ SDE with
the ordinary, non-Karras sigma schedule and writes one self-contained HTML
report containing configuration, metrics, dense/pruned images, and amplified
absolute-difference images.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import os
import random
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from diffusers import DPMSolverSinglestepScheduler, StableDiffusionXLPipeline
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evo_pruning_sdxl as diff_es


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/content/models/hyper_sdxl_4step_471056.safetensors",
    )
    parser.add_argument(
        "--ann-file",
        default="/content/coco/annotations/captions_val2017.json",
    )
    parser.add_argument("--block-id", type=int, default=11)
    parser.add_argument("--num-prompts", type=int, default=12)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument(
        "--output",
        default=None,
        help="One self-contained HTML file. Defaults to /content/block_<id>_all4_report.html",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality used for embedded dense/pruned images.",
    )
    return parser.parse_args()


def scheduler_config(pipe: StableDiffusionXLPipeline) -> None:
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


def load_prompts(path: Path, count: int, seed: int) -> List[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    captions: List[str] = []
    seen = set()
    for ann in raw.get("annotations", []):
        caption = str(ann.get("caption", "")).strip()
        if caption and caption not in seen:
            captions.append(caption)
            seen.add(caption)
    if count > len(captions):
        raise ValueError(f"Requested {count} prompts, but only {len(captions)} were found.")
    return random.Random(seed).sample(captions, count)


def clear_acceleration(unet) -> None:
    if hasattr(unet, "clear_all_accel"):
        unet.clear_all_accel()
        return
    if hasattr(unet, "set_layerdrop"):
        unet.set_layerdrop([])
    elif hasattr(unet, "drop_block_ids"):
        unet.drop_block_ids = set()


def set_static_blockdrop(unet, block_id: int) -> None:
    clear_acceleration(unet)
    if hasattr(unet, "set_layerdrop"):
        unet.set_layerdrop([int(block_id)])
    else:
        unet.drop_block_ids = {int(block_id)}


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device.type if device.type in {"cuda", "cpu"} else "cpu"
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def generate(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    seed: int,
    *,
    block_id: int,
    pruned: bool,
    steps: int,
    cfg_scale: float,
    height: int,
    width: int,
) -> Tuple[Image.Image, float]:
    if pruned:
        set_static_blockdrop(pipe.unet, block_id)
    else:
        clear_acceleration(pipe.unet)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            height=height,
            width=width,
            generator=make_generator(pipe._execution_device, seed),
            output_type="pil",
            return_dict=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return result.images[0].convert("RGB"), elapsed


def image_metrics(dense: Image.Image, pruned: Image.Image) -> Dict[str, float]:
    a = np.asarray(dense, dtype=np.uint8)
    b = np.asarray(pruned, dtype=np.uint8)
    delta = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {
        "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=255)),
        "psnr_db": float(peak_signal_noise_ratio(a, b, data_range=255)),
        "mae_0_1": float(delta.mean() / 255.0),
        "rmse_0_1": float(np.sqrt(np.mean(delta * delta)) / 255.0),
        "max_abs_0_1": float(delta.max() / 255.0),
        "changed_pixel_pct": float(np.any(delta > 0, axis=2).mean() * 100.0),
        "pixels_gt_8_pct": float(np.any(delta > 8, axis=2).mean() * 100.0),
        "pixels_gt_16_pct": float(np.any(delta > 16, axis=2).mean() * 100.0),
    }


def encode_jpeg(image: Image.Image, quality: int) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def difference_image(dense: Image.Image, pruned: Image.Image, gain: float = 4.0) -> Image.Image:
    a = np.asarray(dense, dtype=np.float32)
    b = np.asarray(pruned, dtype=np.float32)
    diff = np.clip(np.abs(a - b) * float(gain), 0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def finite_mean(values: List[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def verdict(mean_ssim: float, min_ssim: float) -> Tuple[str, str]:
    if mean_ssim >= 0.95 and min_ssim >= 0.90:
        return (
            "PASS — strong shared-removal candidate",
            "This screening set supports testing physical deletion and a larger validation run.",
        )
    if mean_ssim >= 0.90 and min_ssim >= 0.80:
        return (
            "PROMISING — expand validation",
            "The block may be removable, but the worst examples still require inspection.",
        )
    return (
        "FAIL FOR HIGH-FIDELITY STATIC REMOVAL",
        "The dense-versus-pruned difference is too large for declaring this block universally safe.",
    )


def build_html(
    *,
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    scheduler: DPMSolverSinglestepScheduler,
    device: torch.device,
    component_counts: Dict[str, int],
) -> str:
    ssims = [r["metrics"]["ssim"] for r in rows]
    psnrs = [r["metrics"]["psnr_db"] for r in rows]
    maes = [r["metrics"]["mae_0_1"] for r in rows]
    dense_times = [r["dense_time_sec"] for r in rows]
    pruned_times = [r["pruned_time_sec"] for r in rows]

    mean_ssim = finite_mean(ssims)
    min_ssim = float(min(ssims))
    verdict_title, verdict_text = verdict(mean_ssim, min_ssim)
    speedup = 1.0 - finite_mean(pruned_times) / max(finite_mean(dense_times), 1e-12)

    summary = {
        "block_id": args.block_id,
        "removed_at_every_denoising_step": True,
        "num_prompts": args.num_prompts,
        "mean_ssim": mean_ssim,
        "median_ssim": float(np.median(ssims)),
        "min_ssim": min_ssim,
        "max_ssim": float(max(ssims)),
        "mean_psnr_db": finite_mean(psnrs),
        "mean_mae_0_1": finite_mean(maes),
        "mean_dense_time_sec": finite_mean(dense_times),
        "mean_pruned_time_sec": finite_mean(pruned_times),
        "measured_speedup_fraction": float(speedup),
        "verdict": verdict_title,
    }

    sorted_rows = sorted(rows, key=lambda item: item["metrics"]["ssim"])
    cards = []
    table_rows = []
    for row in sorted_rows:
        m = row["metrics"]
        table_rows.append(
            "<tr>"
            f"<td>{row['index']}</td><td>{row['seed']}</td>"
            f"<td>{m['ssim']:.6f}</td><td>{m['psnr_db']:.3f}</td>"
            f"<td>{m['mae_0_1']:.6f}</td><td>{m['pixels_gt_16_pct']:.3f}%</td>"
            f"<td>{row['dense_time_sec']:.3f}</td><td>{row['pruned_time_sec']:.3f}</td>"
            f"<td>{html.escape(row['prompt'])}</td></tr>"
        )
        cards.append(
            f"""
<section class="card">
  <h3>Example {row['index']} — seed {row['seed']}</h3>
  <p class="prompt">{html.escape(row['prompt'])}</p>
  <p><b>SSIM:</b> {m['ssim']:.6f} &nbsp; <b>PSNR:</b> {m['psnr_db']:.3f} dB
     &nbsp; <b>MAE:</b> {m['mae_0_1']:.6f} &nbsp; <b>&gt;16 RGB pixels:</b> {m['pixels_gt_16_pct']:.3f}%</p>
  <div class="images">
    <figure><figcaption>Dense</figcaption><img src="data:image/jpeg;base64,{row['dense_b64']}"></figure>
    <figure><figcaption>Block {args.block_id} skipped at every step</figcaption><img src="data:image/jpeg;base64,{row['pruned_b64']}"></figure>
    <figure><figcaption>Absolute difference ×4</figcaption><img src="data:image/png;base64,{row['diff_b64']}"></figure>
  </div>
</section>
"""
        )

    config_json = html.escape(
        json.dumps(
            {
                "arguments": vars(args),
                "scheduler_class": scheduler.__class__.__name__,
                "scheduler_config": dict(scheduler.config),
                "device": str(device),
                "component_parameter_counts": component_counts,
                "summary": summary,
                "per_image": [
                    {
                        "index": row["index"],
                        "seed": row["seed"],
                        "prompt": row["prompt"],
                        "dense_time_sec": row["dense_time_sec"],
                        "pruned_time_sec": row["pruned_time_sec"],
                        **row["metrics"],
                    }
                    for row in rows
                ],
            },
            indent=2,
            default=str,
        )
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SDXL shared block {args.block_id} ablation</title>
<style>
body{{font-family:Arial,sans-serif;background:#f4f5f7;color:#171717;margin:24px;line-height:1.4}}
.card{{background:white;border:1px solid #ddd;border-radius:10px;padding:18px;margin:16px 0}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.metric{{background:#f7f7f8;border-radius:8px;padding:14px}} .value{{font-size:24px;font-weight:700}}
.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}} img{{width:100%;height:auto;border:1px solid #bbb}}
figure{{margin:0}} figcaption{{font-weight:700;margin-bottom:6px}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #ccc;padding:7px;vertical-align:top}} th{{background:#eee}} .prompt{{font-style:italic}}
pre{{white-space:pre-wrap;word-break:break-word;background:#111;color:#eee;padding:14px;overflow:auto}}
@media(max-width:900px){{.images{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>SDXL static shared-block ablation</h1>
<section class="card"><h2>{html.escape(verdict_title)}</h2><p>{html.escape(verdict_text)}</p>
<p>This is a screening result for block <b>{args.block_id}</b> skipped during every one of the {args.steps} denoising steps. The threshold is intentionally strict because the goal is physical, universal removal rather than prompt-specific dynamic skipping.</p></section>
<section class="card"><h2>Aggregate results</h2><div class="metrics">
<div class="metric"><div class="value">{mean_ssim:.6f}</div>Mean SSIM</div>
<div class="metric"><div class="value">{min_ssim:.6f}</div>Worst SSIM</div>
<div class="metric"><div class="value">{finite_mean(psnrs):.3f}</div>Mean PSNR dB</div>
<div class="metric"><div class="value">{finite_mean(maes):.6f}</div>Mean MAE</div>
<div class="metric"><div class="value">{finite_mean(dense_times):.3f}s</div>Mean dense time</div>
<div class="metric"><div class="value">{finite_mean(pruned_times):.3f}s</div>Mean pruned time</div>
<div class="metric"><div class="value">{speedup * 100.0:.2f}%</div>Measured speedup*</div>
</div><p><small>*One-block timing differences are small and may be dominated by GPU timing noise. Quality metrics are the primary result.</small></p></section>
<section class="card"><h2>Configuration</h2>
<p><b>Model:</b> {html.escape(args.model_path)}<br><b>Sampler:</b> DPM++ SDE, normal/non-Karras sigmas<br>
<b>Scheduler:</b> {scheduler.__class__.__name__}, algorithm=sde-dpmsolver++, order=2, midpoint, lower_order_final=True<br>
<b>Steps:</b> {args.steps} &nbsp; <b>CFG:</b> {args.cfg_scale} &nbsp; <b>Resolution:</b> {args.width}×{args.height} &nbsp; <b>Prompts:</b> {args.num_prompts}</p></section>
<section class="card"><h2>Per-image table</h2><table><thead><tr><th>#</th><th>Seed</th><th>SSIM</th><th>PSNR</th><th>MAE</th><th>Pixels &gt;16</th><th>Dense s</th><th>Pruned s</th><th>Prompt</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></section>
{''.join(cards)}
<section class="card"><h2>Embedded machine-readable report</h2><pre>{config_json}</pre></section>
</body></html>"""


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 1024x1024 SDXL report.")

    model_path = Path(args.model_path).expanduser().resolve()
    ann_path = Path(args.ann_file).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)

    output = Path(args.output or f"/content/block_{args.block_id}_all4_report.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    prompts = load_prompts(ann_path, args.num_prompts, args.base_seed)

    print(f"Loading {model_path}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(model_path),
        torch_dtype=torch.float16,
        local_files_only=False,
    )
    scheduler_config(pipe)
    pipe.set_progress_bar_config(disable=True)

    # Keep the VAE in its loaded dtype. SDXL's pipeline performs its own temporary
    # force-upcast and casts latents to post_quant_conv.dtype before decode. For
    # batch-one evaluation this avoids both the prior dtype mismatch and the
    # search-time batch-four memory peak.
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    pipe.to(device)
    pipe.unet = diff_es.attach_pruned_unet(pipe, device)
    clear_acceleration(pipe.unet)

    block_count, wrapper_count = diff_es.count_pruned_blocks(pipe.unet)
    if args.block_id < 0 or args.block_id >= block_count:
        raise ValueError(f"block-id must be in [0,{block_count - 1}], got {args.block_id}")

    component_counts = {
        "unet": sum(p.numel() for p in pipe.unet.parameters()),
        "text_encoder": sum(p.numel() for p in pipe.text_encoder.parameters()),
        "text_encoder_2": sum(p.numel() for p in pipe.text_encoder_2.parameters()),
        "vae": sum(p.numel() for p in pipe.vae.parameters()),
        "prunable_basic_transformer_blocks": block_count,
        "pruned_transformer_wrappers": wrapper_count,
    }

    rows: List[Dict[str, Any]] = []
    print(f"Testing block {args.block_id} at all {args.steps} steps on {args.num_prompts} prompts")
    for index, prompt in enumerate(prompts):
        seed = args.base_seed + index
        dense, dense_time = generate(
            pipe,
            prompt,
            seed,
            block_id=args.block_id,
            pruned=False,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            height=args.height,
            width=args.width,
        )
        pruned, pruned_time = generate(
            pipe,
            prompt,
            seed,
            block_id=args.block_id,
            pruned=True,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            height=args.height,
            width=args.width,
        )
        metrics = image_metrics(dense, pruned)
        rows.append(
            {
                "index": index,
                "seed": seed,
                "prompt": prompt,
                "dense_time_sec": dense_time,
                "pruned_time_sec": pruned_time,
                "metrics": metrics,
                "dense_b64": encode_jpeg(dense, args.jpeg_quality),
                "pruned_b64": encode_jpeg(pruned, args.jpeg_quality),
                "diff_b64": encode_png(difference_image(dense, pruned)),
            }
        )
        print(
            f"[{index + 1:02d}/{args.num_prompts}] SSIM={metrics['ssim']:.6f} "
            f"PSNR={metrics['psnr_db']:.3f} dense={dense_time:.2f}s pruned={pruned_time:.2f}s"
        )

    clear_acceleration(pipe.unet)
    report = build_html(
        args=args,
        rows=rows,
        scheduler=pipe.scheduler,
        device=device,
        component_counts=component_counts,
    )
    output.write_text(report, encoding="utf-8")

    ssims = [row["metrics"]["ssim"] for row in rows]
    print("=" * 80)
    print(f"Mean SSIM:  {np.mean(ssims):.6f}")
    print(f"Worst SSIM: {np.min(ssims):.6f}")
    print(f"One-file report: {output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
