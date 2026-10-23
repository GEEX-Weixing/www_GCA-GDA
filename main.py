"""Command-line entry point for the anonymized GCA-GDA implementation."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from gca_gda.data import DOMAINS
from gca_gda.train import default_config, train_one


TASK_PAIRS = {
    'citation': [('acmv9', 'citationv1'), ('citationv1', 'acmv9'),
                 ('citationv1', 'dblpv7'), ('dblpv7', 'acmv9'),
                 ('dblpv7', 'citationv1')],
    'blog': [('Blog1', 'Blog2'), ('Blog2', 'Blog1')],
    'twitch': [('DE', 'ENGB'), ('ENGB', 'DE'), ('ENGB', 'FR'),
               ('FR', 'DE'), ('FR', 'ENGB')],
    'mag': list(itertools.permutations(DOMAINS['mag'], 2)),
    'elliptic': list(itertools.permutations(DOMAINS['elliptic'], 2)),
}


def main():
    parser = argparse.ArgumentParser(description='GCA-GDA: fixed-config graph adaptation')
    parser.add_argument('--family', choices=[*DOMAINS.keys(), 'all'], default='citation')
    parser.add_argument('--source')
    parser.add_argument('--target')
    parser.add_argument('--all-tasks', action='store_true')
    parser.add_argument('--list-tasks', action='store_true')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=[100])
    parser.add_argument('--config', type=Path, default=None,
                        help='Optional JSON object with overrides of default_config()')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--lambda-j', type=float, default=None)
    parser.add_argument('--scale', type=float, default=None)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Maximum source CE increase in Stage 3')
    parser.add_argument('--evaluate-target', action='store_true',
                        help='Evaluate the fixed final checkpoint only, never select by target labels')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.source or args.target:
        if args.all_tasks or not args.source or not args.target or args.family == 'all':
            parser.error('For one task, specify --family, --source, and --target together')
        if (args.source, args.target) not in TASK_PAIRS[args.family]:
            parser.error('Requested source->target pair is not a supported benchmark task')
        tasks = [(args.family, args.source, args.target)]
    elif args.all_tasks or args.list_tasks:
        groups = DOMAINS if args.family == 'all' else [args.family]
        tasks = [(family, src, tgt) for family in groups for src, tgt in TASK_PAIRS[family]]
    else:
        parser.error('Specify --source and --target, or use --all-tasks / --list-tasks')
    if args.list_tasks:
        for family, src, tgt in tasks:
            print(f'{family}: {src} -> {tgt}')
        return
    cfg = default_config()
    if args.config is not None:
        parsed = json.loads(args.config.read_text(encoding='utf-8'))
        if not isinstance(parsed, dict):
            raise ValueError('--config must contain a JSON object')
        if set(parsed) - set(cfg):
            raise ValueError(f'Unknown config keys: {sorted(set(parsed)-set(cfg))}')
        cfg.update(parsed)
    args_override = {
        'epochs': args.epochs, 'warmup': args.warmup,
        'lambda_j': args.lambda_j, 'scale': args.scale, 'tau': args.tau,
        'source_ce_slack': args.epsilon,
    }
    cfg.update({k: v for k, v in args_override.items() if v is not None})
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for family, src, tgt in tasks:
        for seed in args.seeds:
            print(f'GCA-GDA  {family}  {src} -> {tgt}  seed={seed}', flush=True)
            result = train_one(
                family, src, tgt, seed, config=cfg,
                data_dir=args.data_dir, device=args.device,
                report_target=args.evaluate_target, verbose=args.verbose,
            )
            target_dir = output_root / family
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = target_dir / f'{src}_to_{tgt}_seed{seed}.json'
            filename.write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps({'task': f'{src}->{tgt}', 'seed': seed,
                              'result_file': str(filename),
                              'target_metrics': result['target_metrics']}), flush=True)


if __name__ == '__main__':
    main()
