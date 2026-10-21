"""Opt-in full-model SO(3) GPU checks; launch with Slurm, optionally torchrun.

This is deliberately outside unittest discovery. It reads a few real examples,
uses the production trainer step and DAC, and never writes a checkpoint.
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "egs/rir/EigeNet/EigeNet.json"))
    parser.add_argument("--output", required=True, help="New JSON filename; never overwritten")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--references", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--cases", nargs="+", choices=["disabled", "p0", "p1", "p05"],
                        default=["disabled", "p0", "p1", "p05"])
    parser.add_argument("--mixed-precision", choices=["no", "bf16", "fp16"], default="bf16")
    parser.add_argument("--compile", action="store_true", help="Production dynamic Inductor settings")
    parser.add_argument("--steps-per-case", type=int, default=1)
    parser.add_argument("--benchmark-references", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--data", choices=["real", "synthetic"], default="real")
    args = parser.parse_args()
    if min(args.batch_size, args.steps_per_case, args.warmup, args.repeats,
           args.benchmark_references, *args.references) < 1:
        parser.error("Batch size, counts, warmup and repeats must be positive")
    if Path(args.output).exists():
        parser.error("Output already exists; choose a new filename")
    return args


def load_batch(cfg, batch_size, references, data):
    import torch
    from models.dataset.acousticrooms_dataset import AcousticRooms_Dataset

    if data == "synthetic":
        generator = torch.Generator().manual_seed(cfg.train.random_seed)
        samples = int(cfg.preprocess.sample_rate * cfg.preprocess.duration)
        envelope = torch.exp(-torch.arange(samples) / (samples / 8))
        return {
            "all_ir": 0.05 * torch.randn(batch_size, references + 1, 1, samples,
                                          generator=generator) * envelope,
            "all_cc_src_loc": torch.randn(batch_size, references + 1, 3, generator=generator) * 3,
            "cc_depth_map": torch.randn(batch_size, 3, 256, 512, generator=generator) * 3,
        }, []

    # Use the actual get_batch preprocessing without reading the whole manifest.
    # Any missing or invalid input is reported rather than replaced.
    dataset = AcousticRooms_Dataset.__new__(AcousticRooms_Dataset)
    dataset.cfg = cfg
    dataset.frame_rate = cfg.preprocess.frame_rate
    dataset.sample_rate = cfg.preprocess.sample_rate
    dataset.downsample_rate = dataset.sample_rate // dataset.frame_rate
    dataset.depth_map_root = cfg.dataset.depth_map_root
    dataset.rir_root = cfg.dataset.rir_root
    dataset.metadata_root = cfg.dataset.metadata_root
    dataset.duration = cfg.preprocess.duration
    dataset.reference_count = references
    dataset.depth_resolution = tuple(cfg.model.eigenet_transformer.depth_resol)
    manifests = cfg.dataset.rir_list
    if isinstance(manifests, str):
        manifests = [manifests]
    manifest = manifests[0][0] if isinstance(manifests[0], list) else manifests[0]
    rows, sources = [], []
    random.seed(cfg.train.random_seed)
    with open(manifest, encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= 100:
                break
            item = json.loads(line)
            if len(item["other_src_idx_list"]) < references:
                continue
            batch = dataset.get_batch(item)
            if batch is not None and batch["all_ir"].shape[0] >= references + 1:
                rows.append(batch)
                sources.append({"index": index, "rir_path": item["rir_path"]})
            if len(rows) == batch_size:
                break
    if len(rows) != batch_size:
        raise RuntimeError(f"Could not load {batch_size} real examples with {references} references")
    return {key: torch.stack([torch.as_tensor(row[key]).float() for row in rows])
            for key in ("all_ir", "all_cc_src_loc", "cc_depth_map")}, sources


def subset(batch, references):
    # The last item is always the original target, regardless of reference count.
    return {key: value[:, -(references + 1):].contiguous() if key != "cc_depth_map" else value
            for key, value in batch.items()}


class AuditedRotation:
    def __init__(self, augmentation):
        self.augmentation = augmentation
        self.records = []

    def __call__(self, depth, locations, **kwargs):
        import torch

        old_depth, old_locations = depth.clone(), locations.clone()
        result_depth, result_locations = self.augmentation(depth, locations, **kwargs)
        assert torch.equal(depth, old_depth), "Augmentation mutated input depth"
        assert torch.equal(locations, old_locations), "Augmentation mutated input locations"
        changed = (result_locations != locations).flatten(1).any(1)
        assert torch.equal(result_depth[~changed], depth[~changed]), "Unselected depth changed"
        assert torch.equal(result_locations[~changed], locations[~changed]), "Unselected locations changed"
        assert torch.isfinite(result_depth).all() and torch.isfinite(result_locations).all()
        torch.testing.assert_close(result_locations.norm(dim=-1), locations.norm(dim=-1),
                                   rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(torch.cdist(result_locations, result_locations),
                                   torch.cdist(locations, locations), rtol=2e-5, atol=2e-5)
        probability = self.augmentation.probability
        if not self.augmentation.enabled or probability == 0:
            assert not changed.any(), "Disabled augmentation changed locations"
        elif probability == 1:
            assert changed.all(), "p=1 did not rotate every example"
        self.records.append(changed.cpu().tolist())
        return result_depth, result_locations


def main():
    args = arguments()
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, TorchDynamoPlugin, set_seed
    from models.EigeNet.src.trainer import EigeNetTrainer
    from models.dataset.rotation_augmentation import RotationAugmentation, rotate_geometry, sample_rotations
    from models.layers.utils import OctaveEnergyFeatures
    from models.loss.waveform_loss import loss_fn
    from utils.util import load_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; run through srun/sbatch")
    # Match train.py::cuda_relevant(), including TF32 outside autocast.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    cfg = load_config(args.config)
    plugin = TorchDynamoPlugin(
        backend="inductor", mode="default", dynamic=True, fullgraph=False,
        options={"split_reductions": False},
    ) if args.compile else None
    accelerator = Accelerator(mixed_precision=args.mixed_precision,
                              gradient_accumulation_steps=1,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
                              dynamo_plugin=plugin)
    set_seed(cfg.train.random_seed)
    cfg.train.gradient_accumulation_step = 1
    trainer = EigeNetTrainer.__new__(EigeNetTrainer)
    trainer.cfg, trainer.accelerator = cfg, accelerator
    trainer.sample_rate = cfg.preprocess.sample_rate
    trainer.aligner_activate = cfg.model.eigenet_transformer.aligner.activate
    # Activate every scheduled loss immediately; this is a short training check.
    trainer.step = max(cfg.train.auxiliary_loss_warmup_steps,
                       cfg.train.alignment_loss_warmup_steps, cfg.train.alignment_loss_start_step) + 1
    trainer.batch_count = 0
    trainer.model = trainer._build_model()
    trainer.optimizer = trainer._build_optimizer()
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.model, trainer.optimizer, trainer.scheduler = accelerator.prepare(
        trainer.model, trainer.optimizer, trainer.scheduler)
    trainer.model.train()
    trainer._build_output_model()
    trainer.waveform_loss_fn = loss_fn(
        sr=cfg.preprocess.sample_rate, duration=cfg.preprocess.duration,
        device=accelerator.device, env_duration=cfg.loss.env.duration,
        smooth=cfg.loss.env.smooth, env_window=cfg.loss.env.window,
        env_loss_alpha=cfg.loss.env.alpha,
    ).to(accelerator.device)
    if trainer.aligner_activate:
        trainer.alignment_features = OctaveEnergyFeatures(
            sr=trainer.sample_rate, frame_rate=cfg.preprocess.frame_rate).to(accelerator.device)

    max_references = max(*args.references, args.benchmark_references)
    batch, sources = load_batch(cfg, args.batch_size, max_references, args.data)
    batch = {key: value.to(accelerator.device) for key, value in batch.items()}
    trainable = [p for p in trainer.model.parameters() if p.requires_grad]
    probe = next(p for name, p in trainer.model.named_parameters() if name.endswith("ir_decoder.weight"))
    report = {
        "settings": vars(args), "rank": accelerator.process_index,
        "world_size": accelerator.num_processes, "device": torch.cuda.get_device_name(),
        "torch": torch.__version__, "parameter_count": sum(p.numel() for p in trainable),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "data_sources": sources, "optimizer": "AdamW with configured arguments",
        "scheduler": "constant LambdaLR (no zero-learning-rate warmup)",
        "loss_warmup": "all configured losses active", "cases": [], "benchmark": {},
    }
    rng_values = torch.rand(8, generator=trainer._augmentation_generator(), device=accelerator.device)
    gathered = accelerator.gather(rng_values).reshape(accelerator.num_processes, -1)
    assert torch.equal(rng_values, torch.rand(8, generator=trainer._augmentation_generator(),
                                             device=accelerator.device)), "Generator is not reproducible"
    if accelerator.num_processes > 1:
        assert all(not torch.equal(gathered[0], row) for row in gathered[1:]), "Ranks share augmentation RNG"
    report["rank_rng_samples"] = gathered.cpu().tolist()

    # Check GPU geometry under the same TF32 setting as actual training, against
    # CPU float32 calculations and the independent NumPy experiment convention.
    sys.path.insert(0, str(ROOT / "tests"))
    from test_rotation_augmentation import numpy_rays, numpy_rotate_depth
    import numpy as np
    matrices = sample_rotations(args.batch_size, accelerator.device, trainer._augmentation_generator())
    rotated, points = rotate_geometry(batch["cc_depth_map"], batch["all_cc_src_loc"], matrices)
    cpu_rotated, cpu_points = rotate_geometry(batch["cc_depth_map"].cpu(), batch["all_cc_src_loc"].cpu(), matrices.cpu())
    torch.testing.assert_close(points.cpu(), cpu_points, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(points.norm(dim=-1), batch["all_cc_src_loc"].norm(dim=-1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.cdist(points, points), torch.cdist(batch["all_cc_src_loc"], batch["all_cc_src_loc"]),
                               atol=1e-5, rtol=1e-5)
    scalar = batch["cc_depth_map"][0].double().norm(dim=0).cpu().numpy()
    expected = numpy_rotate_depth(scalar, matrices[0].double().cpu().numpy())
    expected = expected[..., None] * numpy_rays(*scalar.shape)
    observed = rotated[0].permute(1, 2, 0).cpu().numpy()
    # Float32 angular coordinates at 512 columns have subpixel rounding error.
    np.testing.assert_allclose(observed, expected, atol=2e-3, rtol=2e-4)
    report["geometry_precision"] = {
        "cpu_gpu_max_xyz_error": (rotated.cpu() - cpu_rotated).abs().max().item(),
        "numpy_max_xyz_error": float(np.max(np.abs(observed - expected))),
        "max_radius_error": (points.norm(dim=-1) - batch["all_cc_src_loc"].norm(dim=-1)).abs().max().item(),
    }
    del rotated, points, cpu_rotated, cpu_points, matrices

    # A sufficiently large probe checks selected and unselected examples together.
    audit = AuditedRotation(RotationAugmentation(enabled=True, probability=0.5))
    audit(batch["cc_depth_map"][:1].expand(32, -1, -1, -1),
          batch["all_cc_src_loc"][:1].expand(32, -1, -1),
          generator=trainer._augmentation_generator())
    assert any(audit.records[0]) and not all(audit.records[0]), "p=.5 probe did not contain both choices"
    report["mixed_selection_probe"] = audit.records[0]

    current_batch = None
    original_encode = trainer.encode_audio_to_z

    def checked_encode(audio):
        assert torch.equal(audio, current_batch["all_ir"][:, :-1]), "Reference audio changed before DAC"
        return original_encode(audio)

    def check_target(_module, inputs):
        assert torch.equal(inputs[1], current_batch["all_ir"][:, -1]), "Target audio changed before loss"

    target_hook = trainer.waveform_loss_fn.register_forward_pre_hook(check_target)
    trainer.encode_audio_to_z = checked_encode
    choices = {"disabled": (False, 1.0), "p0": (True, 0.0), "p1": (True, 1.0), "p05": (True, 0.5)}

    def step(value):
        # Accelerator.prepare owns model autocast, as in the training loop.
        # Keep DAC decoding and waveform losses in their production precision.
        with accelerator.accumulate(trainer.model):
            result = trainer._train_step(value)
        trainer.batch_count += 1
        trainer.step += 1
        return result

    for references in args.references:
        current_batch = subset(batch, references)
        for case in args.cases:
            enabled, probability = choices[case]
            trainer.rotation_augmentation = AuditedRotation(RotationAugmentation(enabled, probability))
            for repeat in range(args.steps_per_case):
                previous = probe.detach().clone()
                saved_audio = current_batch["all_ir"].clone()
                torch.cuda.synchronize()
                started = time.perf_counter()
                total, losses, _ = step(current_batch)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                assert math.isfinite(total), f"Nonfinite loss: {case}, K={references}"
                assert all(math.isfinite(float(v)) for v in losses.values()), "Nonfinite loss component"
                grads = [p.grad for p in trainable if p.grad is not None]
                assert grads, "No gradients"
                assert all(bool(torch.isfinite(grad).all()) for grad in grads), "Nonfinite gradient"
                assert all(bool(torch.isfinite(p).all()) for p in trainable), "Nonfinite weight"
                delta = (probe.detach() - previous).abs().max().item()
                assert delta > 0, "Decoder weights did not update"
                assert torch.equal(saved_audio, current_batch["all_ir"]), "Audio batch was modified"
                record = {"case": case, "references": references, "repeat": repeat,
                          "losses": losses, "max_weight_delta": delta,
                          "finite_gradient_tensors": len(grads), "seconds": elapsed,
                          "selected": trainer.rotation_augmentation.records[-1],
                          "audio_unchanged": True, "unselected_geometry_unchanged": True}
                report["cases"].append(record)
                print(json.dumps({"rank": accelerator.process_index, **record}), flush=True)
    target_hook.remove()
    trainer.encode_audio_to_z = original_encode
    del previous, saved_audio, grads

    if not args.skip_benchmark:
        current_batch = subset(batch, args.benchmark_references)
        for enabled in (False, True):
            trainer.rotation_augmentation = RotationAugmentation(enabled, 1.0)
            for _ in range(args.warmup):
                step(current_batch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times, loss_values = [], []
            for _ in range(args.repeats):
                accelerator.wait_for_everyone()
                started = time.perf_counter()
                total, _, _ = step(current_batch)
                torch.cuda.synchronize()
                elapsed = torch.tensor(time.perf_counter() - started, device=accelerator.device)
                if accelerator.num_processes > 1:
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                times.append(elapsed.item())
                loss_values.append(total)
                assert math.isfinite(total)
            median = statistics.median(times)
            report["benchmark"]["enabled" if enabled else "disabled"] = {
                "seconds": times, "median_seconds": median,
                "examples_per_second": args.batch_size * accelerator.num_processes / median,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "losses": loss_values,
            }
        baseline, augmented = report["benchmark"]["disabled"], report["benchmark"]["enabled"]
        report["benchmark"]["time_overhead_percent"] = (
            augmented["median_seconds"] / baseline["median_seconds"] - 1) * 100
        report["benchmark"]["allocated_delta_mib"] = (
            augmented["peak_allocated_mib"] - baseline["peak_allocated_mib"])

    report["passed"] = True
    reports = [None] * accelerator.num_processes
    if accelerator.num_processes > 1:
        dist.all_gather_object(reports, report)
    else:
        reports[0] = report
    if accelerator.is_main_process:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"passed": True, "ranks": reports}, stream, indent=2, allow_nan=False)
        print(f"PASS: {path}", flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
