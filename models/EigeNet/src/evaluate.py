"""Shared, deterministic EigeNet evaluation and aggregate-only command line entry.

Run ``python -m models.EigeNet.src.evaluate --help`` from the repository root.
No training state, optimizer or historical FlowMatching interface is loaded.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.dataset.evaluation_dataset import (
    EvaluationDataset, EvaluationShardSampler, collate_evaluation,
)
from models.loss.evaluator import Evaluator, MetricAccumulator


class EigeNetPredictor:
    """Predict target audio without encoding the target; coordinates retain it.

    The caller supplies an unwrapped model: unequal validation shard lengths
    must never enter DDP forward collectives. DAC retains float32 execution,
    matching training; only the CUDA transformer runs with bf16 autocast.
    """
    def __init__(self, model, codec, device):
        self.model = model
        self.codec = codec
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, batch):
        audio = batch['all_ir'].to(self.device, dtype=torch.float32)
        coordinates = batch['all_cc_src_loc'].to(self.device, dtype=torch.float32)
        depth = batch['cc_depth_map'].to(self.device, dtype=torch.float32)
        batch_size, count, channels, length = audio.shape
        if count < 2 or channels != 1 or coordinates.shape[1] != count:
            raise ValueError('Expected references plus one mono target and matching coordinates')
        references = audio[:, :-1].reshape(batch_size * (count - 1), channels, length)
        with torch.autocast(self.device.type, enabled=False):
            z = self.codec.encode_z(references)
        z = z.reshape(batch_size, count - 1, z.shape[-2], z.shape[-1]).transpose(-1, -2)
        with torch.autocast(self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == 'cuda'):
            predicted = self.model(z, depth, coordinates)
        if isinstance(predicted, tuple):
            predicted = predicted[0]
        with torch.autocast(self.device.type, enabled=False):
            return self.codec.decode_z(predicted.transpose(-1, -2).float())


def build_evaluation_loader(cfg, *, manifests=None, batch_size=None,
                            max_references=8, seed=42, rank=0, world_size=1,
                            num_workers=0):
    dataset = EvaluationDataset(cfg, manifests=manifests,
                                max_references=max_references, seed=seed)
    size = cfg.valid.batch_size if batch_size is None else batch_size
    if size <= 0 or num_workers < 0:
        raise ValueError('batch_size must be positive and num_workers nonnegative')
    # DataLoader otherwise consumes the training process's global torch RNG.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=size,
                      sampler=EvaluationShardSampler(dataset, rank, world_size),
                      collate_fn=collate_evaluation, num_workers=num_workers,
                      generator=generator)


def merge_rank_results(accumulator, error=None):
    """Exchange once after independent shards, including empty/error shards.

    An error is communicated before raising so a healthy rank does not wait
    forever in a metrics collective after another rank rejects bad input.
    """
    local = {'error': error, 'state': accumulator.state_dict()}
    if dist.is_available() and dist.is_initialized():
        results = [None] * dist.get_world_size()
        dist.all_gather_object(results, local)
    else:
        results = [local]
    failures = [f'rank {rank}: {item["error"]}' for rank, item in enumerate(results)
                if item['error'] is not None]
    if failures:
        raise RuntimeError('Evaluation failed: ' + '; '.join(failures))
    merged = MetricAccumulator(signal_length=accumulator.signal_length)
    for item in results:
        merged.merge(MetricAccumulator.from_state_dict(item['state']))
    return merged


def evaluate_loader(loader, predictor, evaluator=None):
    evaluator = Evaluator() if evaluator is None else evaluator
    accumulator = MetricAccumulator(signal_length=evaluator.signal_length)
    error = None
    try:
        for packed in loader:
            accumulator.skip(packed['skipped_no_reference'])
            for batch in packed['groups']:
                try:
                    prediction = predictor(batch).detach().float().cpu().numpy()
                    target = batch['all_ir'][:, -1].detach().float().cpu().numpy()
                    detailed = evaluator.evaluate_detailed(target, prediction)
                    accumulator.update(**detailed, reference_counts=batch['reference_counts'])
                except Exception as exc:
                    raise RuntimeError(f'samples {batch["sample_ids"]}: {exc}') from exc
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    return merge_rank_results(accumulator, error)


def load_weights(model, checkpoint):
    """Load the current trainer's plain state dict, strictly, on CPU first."""
    path = Path(checkpoint)
    if path.is_dir():
        path = path / 'pytorch_model.bin'
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def write_summary(summary, output_dir):
    # Serialize before opening: malformed values must not leave partial JSON.
    payload = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
    path = Path(output_dir) / 'summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(payload + '\n')
    return path


def scalar_log_values(summary):
    """Only finite scalar values go to trackers; null/status stay in JSON."""
    values = {key: summary[key] for key in ('samples_evaluated', 'skipped_no_reference')}
    for name, metric in summary['metrics'].items():
        if metric.get('value') is not None:
            values[name] = metric['value']
        for key in ('db', 'valid_count', 'invalid_count'):
            if metric.get(key) is not None:
                values[f'{name}/{key}'] = metric[key]
    for name, diagnostic in summary.get('diagnostics', {}).items():
        if isinstance(diagnostic, dict):
            for key, value in diagnostic.items():
                if value is not None:
                    values[f'diagnostics/{name}/{key}'] = value
        else:
            values[f'diagnostics/{name}'] = diagnostic
    return {key: value for key, value in values.items() if math.isfinite(value)}


def build_predictor(cfg, checkpoint, device):
    # Lazy model imports keep CLI help and CPU metric tests independent of DAC.
    from models.EigeNet.src.model import EigeNetTransformer
    from models.layers.dac_codec import DAC
    model = EigeNetTransformer(duration=cfg.preprocess.duration,
                              frame_rate=cfg.preprocess.frame_rate,
                              cfg=cfg.model.eigenet_transformer)
    load_weights(model, checkpoint)
    model.to(device).eval()
    codec = DAC(cfg.model.DAC.path).to(device).eval()
    if torch.device(device).type == 'cuda':
        model = torch.compile(model, backend='inductor', dynamic=True,
                              options={'split_reductions': False})
    return EigeNetPredictor(model, codec, device)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate current EigeNet; write only summary.json')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True, help='Current weight file or checkpoint directory')
    parser.add_argument('--test-manifest', nargs='+', help='Defaults to config.dataset.test_rir_list')
    parser.add_argument('--output-dir', required=True, help='Must not already contain summary.json')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--reference-count', type=int, default=8, help='Maximum references per target')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    from utils.util import load_config
    cfg = load_config(args.config)
    if (Path(args.output_dir) / 'summary.json').exists():
        raise FileExistsError('Output summary.json already exists; choose a new output directory')
    loader = build_evaluation_loader(cfg, manifests=args.test_manifest,
                                     batch_size=args.batch_size,
                                     max_references=args.reference_count, seed=args.seed,
                                     num_workers=args.num_workers)
    predictor = build_predictor(cfg, args.checkpoint, args.device)
    summary = evaluate_loader(loader, predictor).summary()
    summary['evaluation'] = {'config': args.config, 'checkpoint': args.checkpoint,
                             'manifests': args.test_manifest or cfg.dataset.test_rir_list,
                             'seed': args.seed, 'max_references': args.reference_count,
                             'batch_size': loader.batch_size}
    path = write_summary(summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f'Summary: {path}')
    return summary


if __name__ == '__main__':
    main()
