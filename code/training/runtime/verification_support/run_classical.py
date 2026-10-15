"""Export, calibrate and evaluate the joint BEM extension without neural retraining."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    workspace_data = root.parent / '实验结果及数据'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=workspace_data / 'reference_project',
                        help='Unmodified simulation project belonging to the reference experiment')
    parser.add_argument('--reference-results', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=workspace_data / 'frozen_observations')
    parser.add_argument('--output', type=Path, default=workspace_data / 'classical_core')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--skip-export', action='store_true', help='Use an existing complete raw export')
    args = parser.parse_args()
    ext = Path(__file__).resolve().parent / 'classical_extension'
    project, reference, data, output = (p.resolve() for p in
                                      (args.project, args.reference_results, args.data, args.output))
    commands = []
    if not args.skip_export:
        commands.append([sys.executable, str(ext / 'export_frozen_observations.py'),
                         '--project', str(project), '--reference-results', str(reference),
                         '--output', str(data), '--suite', 'core'])
    if not (output / 'calibration_lock.json').is_file():
        commands.append([sys.executable, str(ext / 'run_classical_cpu.py'), '--data', str(data),
                         '--output', str(output), '--calibrate', '--threads', str(args.threads)])
    commands += [
        [sys.executable, str(ext / 'run_classical_cpu.py'), '--data', str(data), '--output',
         str(output), '--evaluate', '--threads', str(args.threads)],
        [sys.executable, str(ext / 'report_classical.py'), '--classical-output', str(output),
         '--reference-results', str(reference), '--export-manifest', str(data / 'export_manifest.json'),
         '--suite', 'core'],
    ]
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
