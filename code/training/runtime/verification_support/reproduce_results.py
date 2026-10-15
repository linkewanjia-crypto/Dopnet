"""Rebuild completed-study statistics and paper assets from scores; CPU only."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
import shutil


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=root.parent / '实验结果及数据')
    parser.add_argument('--manuscript', type=Path,
                        help='Optional manuscript directory for rebuilding its tables and figures')
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    results = args.results.resolve()
    neural, classical = results / 'dopnet_additions', results / 'classical_core'
    for path in (neural / 'evaluation.json', classical / 'server_report/statistics.json'):
        if not path.is_file():
            parser.error('Missing completed-study file: ' + str(path))
    commands = [
        [sys.executable, str(root / 'aggregate_results.py'), '--output', str(neural),
         '--bootstrap', str(args.bootstrap)],
        [sys.executable, str(root / 'classical_extension/report_classical.py'),
         '--classical-output', str(classical), '--reference-results', str(neural),
         '--archived-scores', '--suite', 'core', '--bootstrap', str(args.bootstrap)],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    if args.manuscript:
        paper = args.manuscript.resolve()
        (paper / 'source_data').mkdir(exist_ok=True)
        for name in ('macro_summary.csv', 'macro_paired_differences.csv', 'statistics.json'):
            shutil.copy2(neural / 'summary' / name, paper / 'source_data' / name)
        for name in ('profile.json', 'receiver_latency.json'):
            shutil.copy2(neural / name, paper / 'source_data' / name)
        commands = [
            [sys.executable, str(paper / 'analysis/build_assets.py')],
            [sys.executable, str(paper / 'analysis/build_classical_assets.py'),
             '--summary', str(classical / 'summary'), '--calibration-lock',
             str(classical / 'calibration_lock.json')],
        ]
        for command in commands:
            subprocess.run(command, check=True)
    print('Completed score-only reproduction; no training, inference or channel generation.')


if __name__ == '__main__':
    main()
