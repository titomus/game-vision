"""GameVision neural feasibility tools.

Run from the repository root:
  python -m neural.experiment train --steps 200
  python -m neural.experiment bench --device cuda --precision fp16
  python -m neural.experiment preview --image screenshot.png --noise 0.07

Nothing is transmitted; no game process is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn

from neural.model import TinyDenoiser, pipeline

DEFAULT_WEIGHTS = Path("models/tiny_denoiser.pt")


def device_for(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible. Vérifiez le pilote NVIDIA et la version CUDA de PyTorch.")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def context_for(device: torch.device, precision: str):
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 est réservé ici à CUDA; utilisez --precision fp32 sur CPU.")
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def synthetic_image(size: int, seed: int) -> Image.Image:
    """Original procedural synthetic textures, no third-party image rights."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(0, 1, size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis)
    hues = rng.uniform(0.18, 0.8, size=3)
    phase = rng.uniform(0, 8)
    freq = rng.uniform(2, 14)
    wave = .10 * np.sin((xx * freq + yy * freq * .3) * math.pi + phase)
    image = np.stack([
        np.clip(hues[i] * (.50 + .45 * xx + .15 * yy) + wave, 0, 1)
        for i in range(3)], axis=-1)
    image = Image.fromarray((image * 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(24):
        a, b = rng.integers(0, size, size=2).tolist()
        c, d = rng.integers(0, size, size=2).tolist()
        color = tuple(int(x) for x in rng.integers(15, 240, size=3))
        if rng.random() < .5:
            draw.line((a, b, c, d), fill=color, width=int(rng.integers(1, 5)))
        else:
            draw.rectangle((min(a, c), min(b, d), max(a, c), max(b, d)),
                           outline=color, width=int(rng.integers(1, 4)))
    return image


def tensor_from_image(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def image_from_tensor(x: torch.Tensor) -> Image.Image:
    arr = (x.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0)
    return Image.fromarray(np.clip(arr + .5, 0, 255).astype(np.uint8), "RGB")


def load_training_patches(folder: Path | None, patch_size: int, count: int, seed: int):
    paths = []
    if folder is not None:
        if not folder.is_dir():
            raise FileNotFoundError(f"Image directory missing: {folder}")
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if not paths:
            raise ValueError(f"No PNG/JPG/WebP images in {folder}")
    patches = []
    rng = random.Random(seed)
    for i in range(count):
        if paths:
            with Image.open(paths[rng.randrange(len(paths))]) as handle:
                img = ImageOps.fit(handle.convert("RGB"), (patch_size, patch_size),
                                   method=Image.Resampling.LANCZOS,
                                   centering=(rng.uniform(.25, .75), rng.uniform(.25, .75)))
        else:
            img = synthetic_image(patch_size, seed * 100003 + i)
        patches.append(tensor_from_image(img))
    return torch.stack(patches), ("user_images" if paths else "procedural_synthetic")


def make_model(width: int, blocks: int, device: torch.device):
    return TinyDenoiser(width=width, blocks=blocks).to(device)


def save_weights(path: Path, model: TinyDenoiser, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "architecture": {"width": model.width, "blocks": model.blocks},
        "model": model.state_dict(),
        "metadata": meta
    }, path)


def restore_weights(path: Path, device: torch.device) -> tuple[TinyDenoiser, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Train it first with: python -m neural.experiment train")
    # weights_only=True rejects arbitrary pickled Python objects from untrusted checkpoints.
    pack = torch.load(path, map_location="cpu", weights_only=True)
    if pack.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    arch = pack["architecture"]
    model = make_model(int(arch["width"]), int(arch["blocks"]), device)
    model.load_state_dict(pack["model"], strict=True)
    model.eval()
    return model, pack.get("metadata", {})


def train(args):
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    if args.steps < 1 or args.patch_size < 16 or args.batch_size < 1:
        raise ValueError("steps, batch-size and patch-size must be positive; patch-size >=16")
    clean_cpu, source = load_training_patches(args.images, args.patch_size,
                                              args.dataset_count, args.seed)
    clean = clean_cpu.to(device)
    model = make_model(args.width, args.blocks, device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    model.train()
    start = time.perf_counter()
    print(f"Training {sum(p.numel() for p in model.parameters())} parameters on {device}; source={source}")
    for step in range(1, args.steps + 1):
        indices = torch.randint(clean.size(0), (args.batch_size,), device=device)
        target = clean.index_select(0, indices)
        noisy = (target + torch.randn_like(target) * args.noise).clamp(0, 1)
        output = model(noisy)
        loss = loss_fn(output, target)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if step == 1 or step == args.steps or step % max(1, args.steps // 5) == 0:
            print(f"step {step:4}/{args.steps} | loss={loss.item():.6f}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    meta = {
        "training_source": source,
        "training_steps": args.steps,
        "noise_sigma": args.noise,
        "seed": args.seed,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "notice": "Synthetic-noise denoiser only. Not DLSS or neural material rendering."
    }
    save_weights(args.weights, model, meta)
    print(f"Checkpoint: {args.weights} | duration={time.perf_counter()-start:.2f}s")
    return meta


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * pct / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def benchmark_case(model, device, precision, height, width, scale, warmup, iterations) -> dict:
    # Fixed GPU-resident input isolates the processing pipeline, not screenshot upload or display.
    input_image = torch.rand((1, 3, height, width), device=device)
    with torch.inference_mode():
        with context_for(device, precision):
            for _ in range(warmup):
                pipeline(model, input_image, scale)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    if device.type == "cuda":
        # Timestamps are GPU elapsed times and exclude Python enqueue overhead.
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(iterations)]
        wall_start = time.perf_counter()
        with torch.inference_mode():
            with context_for(device, precision):
                for begin, end in events:
                    begin.record()
                    pipeline(model, input_image, scale)
                    end.record()
        torch.cuda.synchronize()
        wall_duration = time.perf_counter() - wall_start
        samples = [a.elapsed_time(b) for a, b in events]
        memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        wall_start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(iterations):
                t = time.perf_counter()
                pipeline(model, input_image, scale)
                samples.append((time.perf_counter() - t) * 1000)
        wall_duration = time.perf_counter() - wall_start
        memory = None
    return {
        "resolution": f"{width}x{height}",
        "scale": scale,
        "working_resolution": f"{int(width*scale)}x{int(height*scale)}",
        "gpu_kernel_ms_median": round(statistics.median(samples), 4) if device.type == "cuda" else None,
        "per_call_ms_p50": round(statistics.median(samples), 4),
        "per_call_ms_p95": round(percentile(samples, 95), 4),
        "per_call_ms_p99": round(percentile(samples, 99), 4),
        "wall_throughput_fps": round(iterations / wall_duration, 2),
        "peak_torch_allocated_mib": round(memory, 2) if memory is not None else None,
        "under_6_94ms_shader_only": statistics.median(samples) <= 1000.0 / 144.0,
    }


def parse_resolution(spec: str):
    try:
        width, height = (int(n) for n in spec.lower().split("x"))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"Invalid resolution: {spec}. Example: 1920x1080")
    if width < 16 or height < 16 or width > 7680 or height > 4320:
        raise argparse.ArgumentTypeError("Resolution must be between 16x16 and 7680x4320")
    return (width, height)


def bench(args):
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive; warmup >=0")
    device = device_for(args.device)
    if device.type != "cuda":
        print("WARNING: CPU benchmark; results say NOTHING about RTX 3050 GPU speed.")
    if args.untrained:
        model = make_model(args.width, args.blocks, device).eval()
        meta = {"training_source": "UNTRAINED_IDENTITY", "warning": "Measures network structure only; no visual enhancement"}
    else:
        model, meta = restore_weights(args.weights, device)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "precision": args.precision,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint": "untrained" if args.untrained else str(args.weights),
        "checkpoint_metadata": meta,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "notes": [
            "CUDA event times measure on-GPU pipeline incl. resize, model and clamp.",
            "Input already resides in GPU memory; excludes screen capture, transfer, game rendering, presentation and system latency.",
            "Wall throughput includes Python launch overhead and final synchronization but excludes application/renderer setup.",
            "Meeting 6.94ms alone does not prove ability to run a 144-FPS game concurrently."
        ],
        "results": [],
    }
    for width, height in args.resolutions:
        for scale in args.scales:
            print(f"Benchmark {width}x{height}, scale {scale:.2f}, {gpu}...")
            result = benchmark_case(model, device, args.precision, height, width,
                                    scale, args.warmup, args.iterations)
            report["results"].append(result)
            print(f"  median={result['per_call_ms_p50']:.3f}ms | p95={result['per_call_ms_p95']:.3f}ms | wall {result['wall_throughput_fps']}fps")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["results"][0]))
        writer.writeheader()
        writer.writerows(report["results"])
    print(f"Reports: {args.output} and {csv_path}")
    return report


def preview(args):
    device = device_for(args.device)
    model, meta = restore_weights(args.weights, device)
    if args.image is not None:
        with Image.open(args.image) as handle:
            image = handle.convert("RGB")
    else:
        image = synthetic_image(args.demo_size, args.seed)
    if args.max_edge and max(image.size) > args.max_edge:
        image.thumbnail((args.max_edge, args.max_edge), Image.Resampling.LANCZOS)
    x = tensor_from_image(image).unsqueeze(0).to(device)
    if args.noise:
        torch.manual_seed(args.seed)
        x = (x + args.noise * torch.randn_like(x)).clamp(0, 1)
    with torch.inference_mode():
        with context_for(device, args.precision):
            after = pipeline(model, x, args.scale)
    before_image = image_from_tensor(x[0])
    after_image = image_from_tensor(after[0])
    combined = Image.new("RGB", (before_image.width * 2, before_image.height))
    combined.paste(before_image, (0, 0))
    combined.paste(after_image, (before_image.width, 0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before_image.save(args.output_dir / "before.png")
    after_image.save(args.output_dir / "after.png")
    combined.save(args.output_dir / "compare.png")
    print(f"Preview saved to {args.output_dir.resolve()}")
    print(f"Training source: {meta.get('training_source')}. Denoising is not photorealistic generation.")


def main():
    parser = argparse.ArgumentParser(description="GameVision: honest neural denoising and GPU latency feasibility")
    commands = parser.add_subparsers(dest="command", required=True)
    train_cmd = commands.add_parser("train", help="Train a tiny denoiser on procedural or user-owned images")
    train_cmd.add_argument("--images", type=Path, help="Directory of PNG/JPG/WebP inputs you have rights to use")
    train_cmd.add_argument("--steps", type=int, default=250)
    train_cmd.add_argument("--patch-size", type=int, default=64)
    train_cmd.add_argument("--dataset-count", type=int, default=48)
    train_cmd.add_argument("--batch-size", type=int, default=8)
    train_cmd.add_argument("--noise", type=float, default=0.07)
    train_cmd.add_argument("--lr", type=float, default=0.001)
    train_cmd.add_argument("--width", type=int, default=8)
    train_cmd.add_argument("--blocks", type=int, default=2)
    train_cmd.add_argument("--seed", type=int, default=2026)
    train_cmd.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    train_cmd.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train_cmd.set_defaults(fn=train)
    bench_cmd = commands.add_parser("bench", help="Benchmark GPU-resident inference at different working resolutions")
    bench_cmd.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    bench_cmd.add_argument("--untrained", action="store_true", help="Explicitly benchmark an untrained identity network")
    bench_cmd.add_argument("--width", type=int, default=8, help="Network width only with --untrained")
    bench_cmd.add_argument("--blocks", type=int, default=2, help="Network blocks only with --untrained")
    bench_cmd.add_argument("--resolutions", type=parse_resolution, nargs="+",
                            default=[(960, 540), (1280, 720), (1920, 1080)])
    bench_cmd.add_argument("--scales", type=float, nargs="+", default=[1.0, .75, .50])
    bench_cmd.add_argument("--iterations", type=int, default=40)
    bench_cmd.add_argument("--warmup", type=int, default=10)
    bench_cmd.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    bench_cmd.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    bench_cmd.add_argument("--output", type=Path, default=Path("results/neural_bench.json"))
    bench_cmd.set_defaults(fn=bench)
    prev_cmd = commands.add_parser("preview", help="Make before/after comparison of a denoised image")
    prev_cmd.add_argument("--image", type=Path, help="Optional screenshot to process")
    prev_cmd.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    prev_cmd.add_argument("--output-dir", type=Path, default=Path("results/preview"))
    prev_cmd.add_argument("--demo-size", type=int, default=640)
    prev_cmd.add_argument("--max-edge", type=int, default=1920)
    prev_cmd.add_argument("--noise", type=float, default=0.07, help="Add simulated sensor noise to visibly test denoising; 0 for real screenshot")
    prev_cmd.add_argument("--scale", type=float, default=1.0)
    prev_cmd.add_argument("--seed", type=int, default=10)
    prev_cmd.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    prev_cmd.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    prev_cmd.set_defaults(fn=preview)
    args = parser.parse_args()
    if getattr(args, "scales", None) and any(not 0 < s <= 1 for s in args.scales):
        parser.error("scales must be between 0 and 1")
    if hasattr(args, "noise") and args.noise < 0:
        parser.error("noise must be nonnegative")
    args.fn(args)


if __name__ == "__main__":
    main()
