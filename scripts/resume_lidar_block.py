#!/usr/bin/env python3
"""Resume an interrupted local block with model, Adam and scheduler state.

--inspect validates the saved checkpoint on CPU without starting training.
The original config and already completed blocks remain in place.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from orbvi_gaussian.io import output_path
from orbvi_gaussian.training import confined_environment


def remaining_iterations(checkpoint_step, target_iterations):
    # Nerfstudio 1.1.5 runs range(_start_step, _start_step + max_num_iterations).
    # A checkpoint's step is the last completed zero-based iteration.
    return max(0, target_iterations - (checkpoint_step + 1))


def restore_checkpoint_state(trainer, state):
    """Rebind Adam after Splatfacto replaces Parameters while loading a model.

    Called from Trainer.setup's checkpoint hook, before it builds callbacks.
    Upstream creates optimizers before loading the dynamically sized Gaussians.
    Their Parameter references must be recreated before restoring Adam moments.
    """
    step = int(state['step'])
    trainer.pipeline.load_pipeline(state['pipeline'], step)
    # Splatfacto.load_state_dict unconditionally assigns its inference step30000.
    trainer.pipeline.model.step = step
    trainer._start_step = step + 1
    trainer.optimizers = trainer.setup_optimizers()
    if set(trainer.optimizers.optimizers) != set(state['optimizers']):
        raise RuntimeError('Checkpoint optimizer groups differ from loaded model')
    trainer.optimizers.load_optimizers(state['optimizers'])
    if trainer.config.load_scheduler:
        if set(trainer.optimizers.schedulers) != set(state['schedulers']):
            raise RuntimeError('Checkpoint scheduler groups differ from loaded model')
        trainer.optimizers.load_schedulers(state['schedulers'])
    trainer.grad_scaler.load_state_dict(state['scalers'])
    groups = trainer.pipeline.get_param_groups()
    binding = {}
    for name, optimizer in trainer.optimizers.optimizers.items():
        actual = [p for group in optimizer.param_groups for p in group['params']]
        expected = groups[name]
        if len(actual) != len(expected) or any(a is not b for a, b in zip(actual, expected)):
            raise RuntimeError('Optimizer retains obsolete model Parameters: ' + name)
        shapes = []
        for parameter in actual:
            saved = optimizer.state.get(parameter, {})
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in saved and saved[key].shape != parameter.shape:
                    raise RuntimeError('Restored Adam state shape differs: ' + name + '/' + key)
            shapes.append(list(parameter.shape))
        binding[name] = dict(live_parameter_identity=True, parameter_shapes=shapes)
    return dict(checkpoint_step=step, next_training_step=trainer._start_step,
                model_step=trainer.pipeline.model.step, optimizer_binding=binding,
                scheduler_state_restored=bool(trainer.config.load_scheduler),
                strategy_accumulators='Reinitialized: not stored by upstream checkpoint',
                rng='Reseeded: upstream checkpoint has no image sampler/RNG state')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    out = output_path(args.output)
    os.environ.update(confined_environment())
    import torch
    import yaml
    from nerfstudio.engine.trainer import TrainerConfig

    if (out / 'training_result.json').exists():
        print(json.dumps({'status': 'already_trained', 'output': str(out)}))
        return
    configs = list(out.rglob('config.yml'))
    if len(configs) != 1:
        raise RuntimeError('Expected exactly one saved config in interrupted block')
    config_path = output_path(configs[0])
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    if not isinstance(config, TrainerConfig):
        raise RuntimeError('Invalid local TrainerConfig')
    if output_path(config.get_base_dir()) != config_path.parent:
        raise RuntimeError('Config output location does not match saved run')
    checkpoint_dir = output_path(config.get_checkpoint_dir())
    candidates = sorted(checkpoint_dir.glob('step-*.ckpt'))
    if not candidates:
        raise RuntimeError('Interrupted block has no saved checkpoint; inspect before restarting')
    checkpoint = output_path(candidates[-1])
    digest = hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    required = {'step', 'pipeline', 'optimizers', 'schedulers', 'scalers'}
    if not required.issubset(state):
        raise RuntimeError('Checkpoint lacks model or optimizer state')
    step = int(state['step'])
    target = int(config.max_num_iterations)
    if step != int(checkpoint.stem.split('-')[-1]) or step >= target:
        raise RuntimeError('Checkpoint step does not match saved config')
    remaining = remaining_iterations(step, target)
    report = {'status': 'checkpoint_verified', 'checkpoint': str(checkpoint.relative_to(ROOT)),
              'sha256': digest.hexdigest(), 'checkpoint_step': step,
              'target_iterations': target, 'remaining_iterations': remaining,
              'optimizer_groups': sorted(state['optimizers']),
              'scheduler_groups': sorted(state['schedulers']),
              'note': 'Restores model/optimizers/schedulers/scaler; image shuffle RNG is reseeded.'}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    del state
    if args.inspect:
        return

    # Preserve config.yml's total target and run in its original timestamp folder.
    # Do not use train.main(), which changes the timestamp and swallows Ctrl+C.
    import random
    import numpy as np
    random.seed(config.machine.seed)
    np.random.seed(config.machine.seed)
    torch.manual_seed(config.machine.seed)
    if remaining:
        config.load_dir = checkpoint_dir
        config.load_step = step
        config.load_checkpoint = None
        config.load_scheduler = True
        trainer = config.setup(local_rank=0, world_size=1)
        from types import MethodType
        def load_with_rebound_optimizers(instance):
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            restored = restore_checkpoint_state(instance, saved)
            report['restored_state'] = restored
            with (out/'resume_events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(checkpoint=report,
                    cuda_allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))) + '\n')
            print(json.dumps({'resume_state_verified': restored}, indent=2), flush=True)
        trainer._load_checkpoint = MethodType(load_with_rebound_optimizers, trainer)
        torch.cuda.reset_peak_memory_stats()
        try:
            trainer.setup()
            if trainer._start_step != step + 1:
                raise RuntimeError('Trainer did not restore requested step')
            # Pinned Trainer treats max_num_iterations as an additional count.
            trainer.config.max_num_iterations = remaining
            trainer.train()
        finally:
            trainer.shutdown()
    final_checkpoints = sorted(checkpoint_dir.glob('step-*.ckpt'))
    final_state = torch.load(final_checkpoints[-1], map_location='cpu', weights_only=False)
    if int(final_state['step']) != target - 1:
        raise RuntimeError('Training stopped before target; checkpoint retained without a trained flag')
    count = len(final_state['pipeline']['_model.gauss_params.means'])
    digest = hashlib.sha256()
    with final_checkpoints[-1].open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    del final_state
    result = {'status': 'trained', 'iterations_requested': target,
              'configs': [str(config_path.relative_to(ROOT))],
              'checkpoint': str(final_checkpoints[-1].relative_to(ROOT)),
              'checkpoint_sha256': digest.hexdigest(), 'checkpoint_step': target-1,
              'iterations_completed': target, 'gaussians': count,
              'parameter_origin': 'trained_checkpoint',
              'inria_backend': getattr(config.pipeline.model, 'inria_backend', 'original'),
              'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'gpu_peak_protocol': 'Allocator counters reset before resumed Trainer.setup; resumed segment only',
              'cuda_allocator_config': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
              'resumed_from': report,
              'note': 'Held-out appearance and conditioned geometry still require evaluation'}
    if (out/'refinement.json').is_file():
        result['refinement'] = json.loads((out/'refinement.json').read_text())
    if (out/'launch.json').is_file():
        launch = json.loads((out/'launch.json').read_text())
        result['renderer_provenance'] = launch.get('renderer_provenance')
        result['initialization'] = {key: launch.get(key) for key in
            ('initialization_origin', 'initialization_sha256', 'warm_ply', 'warm_opacity_cap')}
    temporary = out / 'training_result.json.tmp'
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(out / 'training_result.json')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
