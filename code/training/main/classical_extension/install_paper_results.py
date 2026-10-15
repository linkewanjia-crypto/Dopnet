"""Install completed, verified classical tables into the manuscript source tree."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path,
                        help='The summary folder produced by report_classical.py')
    parser.add_argument('--manuscript', required=True, type=Path,
                        help='Manuscript root, containing source/ and figures/')
    args = parser.parse_args()
    report, paper = args.report.resolve(), args.manuscript.resolve()
    status = json.loads((report / 'statistics.json').read_text(encoding='utf-8'))
    if (status.get('paper_ready') is not True
            or status.get('completed_verified_cells') != status.get('expected_cells')
            or status.get('suite') not in ('core', 'all')):
        raise RuntimeError('The declared classical comparison has not been completed and verified')
    if not (paper / 'source' / 'DOPNet_Manuscript.tex').is_file():
        raise RuntimeError('Use the supplied manuscript source tree')
    main_source = (paper / 'source' / 'DOPNet_Manuscript.tex').read_text(encoding='utf-8')
    builder = paper / 'analysis' / 'build_classical_assets.py'
    if '\\label{tab:joint-bem}' in main_source and builder.is_file():
        # The revised paper already contains the curated comparison. Rebuild its
        # data assets rather than appending a second copy of the same subsection.
        subprocess.run([sys.executable, str(builder), '--summary', str(report),
                        '--calibration-lock', str(report.parent / 'calibration_lock.json')], check=True)
        print('Rebuilt existing joint-BEM paper assets. Recompile both PDFs.')
        return
    required = ['paper_insertion.tex', 'classical_nmse.tex', 'classical_ber.tex',
                'native_receiver_insertion.tex', 'classical_native_ber.tex',
                'macro_summary.csv', 'macro_paired_differences.csv', 'statistics.json']
    missing = [name for name in required if not (report / name).is_file()]
    if missing:
        raise RuntimeError('Missing completed report assets: ' + ', '.join(missing))
    # If the reporter supplies bindings, verify them before touching the paper.
    for name, expected in status.get('report_asset_sha256', {}).items():
        path = (report / name).resolve()
        try:
            path.relative_to(report)
        except ValueError as error:
            raise RuntimeError('Unsafe report asset path: ' + name) from error
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError('Report asset changed after reporting: ' + name)
    target = paper / 'source' / 'classical_results'
    target.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(report / name, target / name)
    for path in report.glob('classical_*.pdf'):
        shutil.copy2(path, paper / 'figures' / path.name)
    for path in report.glob('classical_*.png'):
        shutil.copy2(path, paper / 'figures' / path.name)
    shutil.copy2(report / 'paper_insertion.tex', paper / 'source' / 'classical_results_section.tex')
    shutil.copy2(report / 'native_receiver_insertion.tex',
                 paper / 'source' / 'classical_native_results_section.tex')
    supplement = paper / 'source' / 'DOPNet_Supplement.tex'
    text = supplement.read_text(encoding='utf-8')
    insertion = '\\IfFileExists{classical_native_results_section.tex}{\\input{classical_native_results_section.tex}}{}'
    if insertion not in text:
        text = text.replace('\\end{document}', insertion + '\n\\end{document}')
        supplement.write_text(text, encoding='utf-8')
    record = dict(suite=status['suite'], cells=status['expected_cells'],
                  statistics_sha256=sha256(report / 'statistics.json'),
                  main_section_sha256=sha256(paper / 'source' / 'classical_results_section.tex'),
                  inserted_from=str(report))
    (paper / 'CLASSICAL_RESULTS_INSTALLED.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('Installed actual classical results. Recompile both PDFs and review the final layout.')

if __name__ == '__main__':
    main()
