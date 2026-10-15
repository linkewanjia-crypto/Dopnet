"""Emit fixed 40-epoch equal-budget comparison assets from complete frozen scores."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

if not __package__:
    _package_dir = Path(__file__).resolve().parent
    sys.path[:] = [value for value in sys.path if Path(value or '.').resolve() != _package_dir]
    sys.path.insert(0, str(_package_dir.parent))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--training-output', type=Path, required=True)
    a = ap.parse_args()
    from budget_matched_baselines.verify_author_delivery import (
        check_budget_marker, check_history, checkpoint_metadata, verify_prefix_provenance,
        signed, RECIPE, CLAIM, MAX_EPOCHS, TERMINATION,
    )
    out = a.output.resolve()
    done = json.loads((out / 'COMPLETE.json').read_text())
    ev = json.loads((out / 'evaluation.json').read_text())
    statistics = json.loads((out / 'summary/statistics.json').read_text())
    for obj in (ev, statistics):
        expected = hashlib.sha256(json.dumps({k: v for k, v in obj.items() if k != 'signature'},
                                             sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if obj.get('signature') != expected:
            raise RuntimeError('Invalid signed evaluation/statistics')
    if (done['status'] != 'complete' or done['signature'] != ev['signature']
            or done['cells'] != len(ev['cells']) or statistics['evaluation_signature'] != ev['signature']):
        raise RuntimeError('Only complete, bound evaluations may produce report assets')
    recipe_path = a.training_output / 'recipe_manifest.json'
    recipe = signed(json.loads(recipe_path.read_text(encoding='utf-8')), 'equal-budget training plan')
    if (recipe.get('recipe') != RECIPE or recipe.get('claim') != CLAIM
            or ev.get('training_recipe_sha256') != sha(recipe_path)
            or ev.get('training_claim') != CLAIM or ev.get('training_max_epochs') != MAX_EPOCHS
            or ev.get('training_termination') != TERMINATION):
        raise RuntimeError('Report requires the signed 40-epoch equal-budget plan and its bound scores')
    training_records = []
    for seed in (42, 43, 44):
        folder = a.training_output / 'checkpoints' / f'fortitran_budget_matched_seed{seed}'
        marker = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
        check_budget_marker(marker, seed, recipe['signature'])
        if (marker.get('selected_checkpoint_sha256') != sha(folder / 'model.pt')
                or marker.get('last_checkpoint_sha256') != sha(folder / 'last.pt')):
            raise RuntimeError('Training checkpoint changed before reporting: seed' + str(seed))
        with (folder / 'history.csv').open(encoding='utf-8-sig', newline='') as stream:
            history = list(csv.DictReader(stream))
        selected = checkpoint_metadata((folder / 'model.pt').read_bytes(), folder.name + '.model.pt')
        last = checkpoint_metadata((folder / 'last.pt').read_bytes(), folder.name + '.last.pt')
        check_history(history, selected, last, marker, folder.name)
        training_records.append((seed, selected['cfg'], [selected, last], history))
    for cell in ev['cells']:
        path = out / 'cells' / (cell['cell_id'] + '.npz')
        if statistics['cell_file_sha256'][cell['cell_id']] != sha(path):
            raise RuntimeError('Score file changed after statistics')
    for name, expected in statistics['summary_csv_sha256'].items():
        if sha(out / 'summary' / name) != expected:
            raise RuntimeError('Summary CSV changed after statistics')
    with (out / 'summary/macro_summary.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    labels = {'dopnet': 'DOP-Net', 'fortitran': 'FortiTran (common recipe)',
              'fortitran_budget_matched': 'FortiTran (equal budget)'}
    selected = [r for r in rows if r['doppler_hz'] == 'all' and r['snr_db'] == 'all'
                and ((r['study'] == 'nmse' and r['metric'] == 'nmse' and r['scope'] in ('gpp_seen', 'gpp_heldout', 'gpp_high_mobility'))
                     or (r['study'] == 'receiver' and r['metric'] == 'ber_diagonal' and r['scope'] in ('gpp_seen', 'gpp_heldout')))]
    dest = out / 'paper_assets'
    dest.mkdir(exist_ok=True)
    lines = ['\\begin{table*}[t]', '\\centering', '\\scriptsize',
             '\\caption{FortiTran equal-budget comparison on identical test frames. Values are averaged over training seeds 42, 43, and 44; intervals are 95\\% training-seed intervals. The source-guided equal-budget run uses Adam, an MSE objective, effective batch 512, exponential learning-rate decay, and 40 epochs without early stopping. All methods retain the task geometry and common fixed validation observations.}',
             '\\label{tab:fortitran_budget_matched}', '\\begin{tabular}{llllr}',
             '\\toprule', 'Metric & Modulation & Scope & Method & Mean [95\\% CI] \\\\', '\\midrule']
    for r in sorted(selected, key=lambda x: (x['metric'], x['scope'], x['method'])):
        if r['metric'] == 'nmse':
            keys = ('mean_db', 'seed_ci95_lo_db', 'seed_ci95_hi_db')
            values = [f'{float(r[k]):.2f}' if r.get(k) else '--' for k in keys]
            metric = 'NMSE (dB)'
        else:
            keys = ('mean', 'seed_ci95_lo', 'seed_ci95_hi')
            values = [f'{float(r[k]):.3g}' if r.get(k) else '--' for k in keys]
            metric = 'BER'
        lines.append(' & '.join((metric, r['modulation'].upper(), r['scope'].replace('_', '\\_'), labels[r['method']],
                                  f'{values[0]} [{values[1]}, {values[2]}]')) + ' \\\\')
    lines.extend(('\\bottomrule', '\\end{tabular}', '\\end{table*}', ''))
    (dest / 'fortitran_budget_matched_table.tex').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    for ax, scope in zip(axes, ('gpp_seen', 'gpp_heldout')):
        for method, label in labels.items():
            points = sorted([r for r in rows if r['method'] == method and r['scope'] == scope
                             and r['study'] == 'nmse' and r['metric'] == 'nmse' and r['modulation'] == 'qpsk' and r['snr_db'] == '25.0'
                             and r['doppler_hz'] != 'all'], key=lambda r: float(r['doppler_hz']))
            if not points:
                continue
            x = [float(r['doppler_hz']) for r in points]
            y = [float(r['mean_db']) for r in points]
            ax.plot(x, y, marker='o', markersize=3, label=label)
            if all(r.get('seed_ci95_lo_db') and r.get('seed_ci95_hi_db') for r in points):
                ax.fill_between(x, [float(r['seed_ci95_lo_db']) for r in points],
                                [float(r['seed_ci95_hi_db']) for r in points], alpha=.12)
        ax.set_title('Seen 3GPP' if scope == 'gpp_seen' else 'Held-out 3GPP')
        ax.set_xlabel('Maximum Doppler shift (Hz)')
        ax.set_ylabel('NMSE (dB)')
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.savefig(dest / 'fortitran_budget_matched.pdf')
    fig.savefig(dest / 'fortitran_budget_matched.png', dpi=250)
    plt.close(fig)
    files = {}
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.csv', '.npz', '.tex', '.pdf', '.png'):
            files['evaluation/' + path.relative_to(out).as_posix()] = path
    for path in sorted(a.training_output.rglob('*')):
        relative = path.relative_to(a.training_output)
        if relative.parts[0] in ('evaluation', 'queue') or path.name == 'start.log':
            continue
        if path.is_file() and (relative.parts[0] == 'prefix_provenance'
                               or path.suffix in ('.json', '.yaml', '.csv', '.pt', '.log')):
            files['training/' + path.relative_to(a.training_output).as_posix()] = path
    # Read each file exactly once. A controller can finish logging after this
    # report returns; the archived evidence is a consistent byte snapshot.
    snapshots = {name: path.read_bytes() for name, path in files.items()}
    manifest = {'schema': 1, 'status': 'complete', 'evaluation_signature': ev['signature'],
                'training_claim': CLAIM, 'training_termination': TERMINATION, 'training_max_epochs': MAX_EPOCHS,
                'files': {name: hashlib.sha256(data).hexdigest() for name, data in snapshots.items()}}
    package = out / 'budget_matched_results.zip'
    temporary = package.with_suffix('.tmp')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as z:
        for name, data in snapshots.items():
            z.writestr(name, data)
        z.writestr('DELIVERY_MANIFEST.json', json.dumps(manifest, indent=2))
    with ZipFile(temporary) as z:
        for seed, cfg, states, history in training_records:
            verify_prefix_provenance(z, manifest['files'], recipe, cfg, states, history, seed)
    temporary.replace(package)
    package.with_suffix('.zip.sha256').write_text(sha(package) + '  ' + package.name + '\n')
    print('Source-guided equal-budget paired comparison, figure, table and result package complete', flush=True)


if __name__ == '__main__':
    main()
