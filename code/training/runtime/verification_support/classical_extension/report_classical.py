#!/usr/bin/env python3
"""Report verified CPU classical scores against the original test-frame scores.

No training, replay or inference occurs in this program. The manuscript fragment
is produced only for a complete predeclared suite with a locked calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_classical_cpu import code_hashes, object_hash, read_complete, validated_lock

SEEDS = (42, 43, 44)
LEARNED = ('dopnet', 'slow_time', 'dopnet_lite', 'fortitran')
OLD_CLASSICAL = ('ls_interp', 'lmmse_wiener', 'dpsbem_ls')
NEW_CLASSICAL = ('oce_bem_joint_ridge', 'dpss_bem_joint_ridge')
SEEN = ('tdl_b', 'tdl_c', 'tdl_d', 'cdl_b', 'cdl_c', 'cdl_d')
HELDOUT = ('tdl_a', 'tdl_e', 'cdl_a', 'cdl_e')
LABELS = {
    'dopnet': 'DOP-Net', 'slow_time': 'Slow-time control',
    'dopnet_lite': 'DOP-Net-Lite', 'fortitran': 'FortiTran adapter',
    'ls_interp': 'LS interpolation', 'lmmse_wiener': 'Separable Wiener',
    'dpsbem_ls': 'Pilot-grid DPS-BEM LS',
    'oce_bem_joint_ridge': 'Joint OCE-BEM ridge',
    'dpss_bem_joint_ridge': 'Joint DPSS-BEM ridge',
}
ORDER = OLD_CLASSICAL + NEW_CLASSICAL + LEARNED
CI_CLASSICAL = 'paired test-cluster percentile bootstrap; no training seeds'
CI_LEARNED = 'joint training-seed and paired test-cluster percentile bootstrap'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scalar(z, name):
    if name not in z:
        raise ValueError('Missing NPZ field: ' + name)
    value = z[name]
    if value.shape != ():
        raise ValueError('Expected scalar NPZ field: ' + name)
    return str(value.item())


class Reference:
    """Read extracted results or their ZIP without extracting checkpoints/cache."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = None
        if self.path.is_dir():
            candidates = [self.path, self.path / 'results' / 'dopnet_additions',
                          self.path / 'DOPNet_Submission_Experiments' / 'results' / 'dopnet_additions']
            candidates = [p for p in candidates if (p / 'evaluation.json').is_file()]
            if len(candidates) != 1:
                raise ValueError('Reference folder must identify exactly one evaluation.json')
            self.root = candidates[0]
            self.evaluation = read_json(self.root / 'evaluation.json')
            self.experiment = read_json(self.root / 'experiment.json')
        elif zipfile.is_zipfile(self.path):
            self.archive = zipfile.ZipFile(self.path)
            matches = [n for n in self.archive.namelist()
                       if n.endswith('/evaluation.json') or n == 'evaluation.json']
            preferred = [n for n in matches if '/results/dopnet_additions/' in '/' + n]
            matches = preferred or matches
            if len(matches) != 1:
                raise ValueError('Reference ZIP must identify exactly one evaluation.json')
            self.root = matches[0][:-len('evaluation.json')]
            self.evaluation = json.loads(self.archive.read(matches[0]).decode('utf-8-sig'))
            self.experiment = json.loads(self.archive.read(self.root + 'experiment.json').decode('utf-8-sig'))
        else:
            raise ValueError('Reference results must be an extracted results directory or ZIP')
        self.metadata = {c['cell_id']: c for c in self.evaluation['cells']}
        for name, value in (('evaluation', self.evaluation), ('experiment', self.experiment)):
            unsigned = {k: v for k, v in value.items() if k != 'signature'}
            # Historical protocol used ASCII-escaped canonical JSON, including any saved path strings.
            calculated = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'),
                                                    allow_nan=False).encode('utf-8')).hexdigest()
            if value.get('signature') != calculated:
                raise ValueError('Invalid original ' + name + ' signature')
        if self.evaluation.get('experiment_signature') != self.experiment['signature']:
            raise ValueError('Original evaluation and experiment signatures disagree')
        if len(self.metadata) != len(self.evaluation['cells']):
            raise ValueError('Duplicate reference cell IDs')
        self.routes = self.evaluation['routes']
        self.signature = self.evaluation['signature']
        self.indices = {}
        for method in LEARNED + OLD_CLASSICAL:
            idx = [i for i, r in enumerate(self.routes) if r['method'] == method]
            expected = list(SEEDS) if method in LEARNED else [None]
            if [self.routes[i]['training_seed'] for i in idx] != expected:
                raise ValueError('Reference routes/seeds mismatch for ' + method)
            self.indices[method] = idx

    def cell(self, cell_id):
        if self.archive is not None:
            content = self.archive.read(self.root + 'cells/' + cell_id + '.npz')
            source = io.BytesIO(content)
        else:
            source = self.root / 'cells' / (cell_id + '.npz')
        with np.load(source, allow_pickle=False) as z:
            if scalar(z, 'signature') != self.signature:
                raise ValueError('Stale reference NPZ signature: ' + cell_id)
            meta = json.loads(scalar(z, 'meta'))
            if meta != self.metadata[cell_id]:
                raise ValueError('Reference metadata mismatch: ' + cell_id)
            return {name: z[name].copy() for name in z.files}

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def verify_provenance(self, provenance):
        if (provenance.get('reference_evaluation_signature') != self.signature
                or provenance.get('reference_experiment_signature') != self.experiment['signature']
                or provenance.get('reference_cells') != len(self.metadata)
                or provenance.get('reference_samples_per_cell') != 512
                or provenance.get('reference_batch_size') != 8):
            raise ValueError('Reference provenance differs from the frozen score archive')
        fingerprints = {}
        for cid in self.metadata:
            fingerprints[cid] = str(self.cell(cid)['realization_sha256'].item())
        source = dict(evaluation_signature=self.signature,
                      experiment_signature=self.experiment['signature'],
                      waveform_keys=('input', 'add_input', 'add_input_slow', 'rx_data', 'target_h', 'tx_label'),
                      realization_sha256=fingerprints)
        expected = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':'),
                                             allow_nan=False).encode('utf8')).hexdigest()
        if provenance.get('reference_realizations_sha256') != expected:
            raise ValueError('Original full-grid waveform fingerprint provenance differs')


def resampling_weights(cluster_ids, repetitions, rng):
    ids, counts = np.unique(cluster_ids, return_counts=True)
    if len(ids) < 32 or len(set(counts.tolist())) != 1:
        raise ValueError('At least 32 equal-size independent generation clusters required')
    weights = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=repetitions)
    return ids, weights


def metric_samples(values, bits, cluster, weights, ids):
    """Ratio of resampled totals for BER; mean linear frame NMSE otherwise."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(cluster) or not np.isfinite(data).all():
        raise ValueError('Invalid per-route frame values')
    if np.any(data < 0):
        raise ValueError('Negative metric values')
    if bits is None:
        per_cluster = np.stack([data[:, cluster == c].mean(axis=1) for c in ids], axis=1)
        point = per_cluster.mean(axis=1)
        boot = weights @ per_cluster.T / len(ids)
    else:
        total = np.asarray(bits, dtype=np.float64)
        if total.shape != data.shape or np.any(total <= 0) or np.any(data > total):
            raise ValueError('Invalid BER error/bit totals')
        errors_c = np.stack([data[:, cluster == c].sum(axis=1) for c in ids], axis=1)
        bits_c = np.stack([total[:, cluster == c].sum(axis=1) for c in ids], axis=1)
        point = data.sum(axis=1) / total.sum(axis=1)
        boot = (weights @ errors_c.T) / (weights @ bits_c.T)
    return point, boot


def final_draws(boot, seed_draws):
    if boot.shape[1] == 1:
        return boot[:, 0]
    if boot.shape[1] != 3:
        raise ValueError('Learned routes must contain exactly three training seeds')
    return boot[np.arange(len(boot))[:, None], seed_draws].mean(axis=1)


def interval_row(context, method, metric, point, boot, seed_draws):
    mean = float(np.mean(point))
    draws = final_draws(boot, seed_draws)
    low, high = np.quantile(draws, [.025, .975])
    row = dict(context, method=method, estimator=LABELS.get(method, method), metric=metric,
               n_training_seeds=3 if method in LEARNED else 0, mean=mean,
               ci95_lo=float(low), ci95_hi=float(high),
               interval=CI_LEARNED if method in LEARNED else CI_CLASSICAL)
    if metric == 'nmse':
        if mean <= 0 or low <= 0:
            raise ValueError('Nonpositive NMSE cannot be expressed in dB')
        row.update(mean_db=float(10 * np.log10(mean)),
                   ci95_lo_db=float(10 * np.log10(low)),
                   ci95_hi_db=float(10 * np.log10(high)))
    else:
        row['zero_error_observation'] = bool(mean == 0)
    if method in LEARNED:
        row.update({f'seed{s}': float(v) for s, v in zip(SEEDS, point)})
    return row, draws


def paired_row(context, method, metric, point, boot, reference_point, reference_boot, seed_draws):
    ref_draws = final_draws(reference_boot, seed_draws)
    method_draws = final_draws(boot, seed_draws)
    delta = method_draws - ref_draws
    low, high = np.quantile(delta, [.025, .975])
    row = dict(context, comparator=method, reference='dopnet', metric=metric,
               paired_difference=float(np.mean(point) - np.mean(reference_point)),
               ci95_lo=float(low), ci95_hi=float(high),
               interpretation='positive = comparator has a larger error',
               interval='joint seed/paired cluster bootstrap; pointwise 95% interval')
    if metric == 'nmse':
        gap = 10 * np.log10(method_draws / ref_draws)
        lo, hi = np.quantile(gap, [.025, .975])
        row.update(paired_gap_db=float(10 * np.log10(np.mean(point) / np.mean(reference_point))),
                   paired_gap_ci95_lo_db=float(lo), paired_gap_ci95_hi_db=float(hi))
    return row


def is_nominal(meta):
    return (int(meta['cp_len']) == 16 and int(meta['pilot_spacing']) == 4
            and math.isclose(float(meta['delay_spread_ns']), 300.0))


def memberships(meta):
    """The main matched grid excludes pilot/delay/geometry sensitivity cells."""
    if not is_nominal(meta) or meta['channel'] not in ('doubly_selective',) + SEEN + HELDOUT:
        return []
    studies = meta.get('studies', [])
    if 'nmse' in studies and meta['modulation'] == 'qpsk':
        family = 'nmse_qpsk'
    elif ('receiver' in studies and meta['modulation'] == '16qam'
          and math.isclose(float(meta['snr_db']), 20.0)):
        family = 'receiver_16qam_20db'
    else:
        return []
    scopes = ['selected_dynamic', 'dynamic_fd_%g' % float(meta['doppler_hz'])]
    if meta['channel'] in SEEN:
        scopes.append('gpp_seen')
    if meta['channel'] in HELDOUT:
        scopes.append('gpp_heldout')
        scopes.append('heldout_fd_%g' % float(meta['doppler_hz']))
    return [(family, scope) for scope in scopes]


def npz_matrix(z, key, routes, frames):
    if key not in z:
        raise ValueError('Missing score field: ' + key)
    arr = np.asarray(z[key])
    if arr.shape != (routes, frames):
        raise ValueError(f'{key}: expected {(routes, frames)}, got {arr.shape}')
    return arr


def bindings(complete):
    value = complete.get('cell_files', complete.get('cells', {}))
    if isinstance(value, dict):
        return {str(k): (v.get('sha256') if isinstance(v, dict) else str(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return {str(v['cell_id']): str(v['sha256']) for v in value}
    return {}


def validate_core(meta_list):
    expected = {(ch, fd, mod, snr) for ch in ('doubly_selective',) + SEEN + HELDOUT
                for fd in (800.0, 1200.0, 1600.0)
                for mod, snr in (('qpsk', 15.0), ('qpsk', 25.0), ('16qam', 20.0))}
    actual = {(m['channel'], float(m['doppler_hz']), m['modulation'], float(m['snr_db']))
              for m in meta_list}
    if len(meta_list) != 99 or actual != expected or not all(is_nominal(m) for m in meta_list):
        raise ValueError('Core suite must be the predeclared 99-cell high-mobility matrix')


def format_value(row):
    if row is None:
        return '--'
    if row['metric'] == 'nmse':
        keys, factor = ('mean_db', 'ci95_lo_db', 'ci95_hi_db'), 1
    else:
        keys, factor = ('mean', 'ci95_lo', 'ci95_hi'), 100
    vals = [row[k] * factor for k in keys]
    return f'{vals[0]:.3f} [{vals[1]:.3f}, {vals[2]:.3f}]'


def write_tables(dest, macro, suite):
    lookup = {(r['family'], r['scope'], r['metric'], r['method']): r
              for r in macro if r['suite'] == suite}
    groups = ('selected_dynamic', 'gpp_seen', 'gpp_heldout')
    header = ['Selected dynamic', 'Seen 3GPP', 'Held-out 3GPP']
    common_note = ('Means [pointwise 95\\% bootstrap intervals]. '
                   'Linear NMSE and bit-count BER are averaged within each channel, then equally across channels. '
                   'Intervals use shared eight-frame generation clusters and the three learned-model seeds; '
                   'classical estimators have no training seeds.')
    grid_name = 'Matched high-mobility' if suite == 'core' else 'Matched dynamic-channel'
    specs = [
        ('nmse_qpsk', 'nmse', 'classical_nmse', grid_name + ' NMSE (dB)', groups, header),
        ('receiver_16qam_20db', 'ber_diagonal', 'classical_ber',
         grid_name + ' 16-QAM BER at 20 dB (\\%) with the common diagonal detector', groups, header),
        ('receiver_16qam_20db', 'ber_native', 'classical_native_ber',
         grid_name + ' 16-QAM BER at 20 dB (\\%) with estimator-specific ICI-aware detection. '
         'The new classical estimators use full occupied-band ICI operators; learned estimators '
         'use their original five-band ICI interface', groups, header),
    ]
    tables = []
    for family, metric, name, caption, scopes, titles in specs:
        available = [m for m in ORDER if any((family, s, metric, m) in lookup for s in scopes)]
        if not available:
            continue
        lines = ['\\begin{table*}[t]', '\\centering', '\\scriptsize',
                 '\\caption{' + caption + '. ' + common_note + '}',
                 '\\label{tab:' + name + '}', '\\begin{tabular}{lccc}', '\\toprule',
                 'Estimator & ' + ' & '.join(titles) + ' \\\\', '\\midrule']
        for method in available:
            entries = [format_value(lookup.get((family, scope, metric, method))) for scope in scopes]
            lines.append(LABELS[method] + ' & ' + ' & '.join(entries) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
        (dest / (name + '.tex')).write_text('\n'.join(lines), encoding='utf-8')
        tables.append(name)
    return tables


def make_plots(dest, macro, suite):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Install matplotlib, or use the supplied conda Python, to render scientific plots') from exc
    selected = [r for r in macro if r['suite'] == suite]
    figures = []
    for family, metric, basename, ylabel in (
        ('nmse_qpsk', 'nmse', 'classical_nmse', 'NMSE (dB)'),
        ('receiver_16qam_20db', 'ber_diagonal', 'classical_ber', 'Common-detector BER (%)'),
        ('receiver_16qam_20db', 'ber_native', 'classical_native_ber', 'Estimator-specific ICI-aware BER (%)'),
    ):
        rows = [r for r in selected if r['family'] == family and r['metric'] == metric]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
        for ax, scope, title in zip(axes, ('selected_dynamic', 'gpp_seen', 'gpp_heldout'),
                                   ('Selected dynamic', 'Seen 3GPP', 'Held-out 3GPP')):
            by_method = {r['method']: r for r in rows if r['scope'] == scope}
            methods = [m for m in ORDER if m in by_method]
            for index, method in enumerate(methods):
                row = by_method[method]
                keys, factor = (('mean_db', 'ci95_lo_db', 'ci95_hi_db'), 1) if metric == 'nmse' else (
                    ('mean', 'ci95_lo', 'ci95_hi'), 100)
                mean, low, high = [row[k] * factor for k in keys]
                color = '#bc3c29' if method == 'dopnet' else '#0072b2' if method in NEW_CLASSICAL else '#777777'
                ax.plot(index, mean, 'o', color=color, markersize=4)
                ax.vlines(index, low, high, color=color, linewidth=1)
                ax.hlines([low, high], index - .09, index + .09, color=color, linewidth=1)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([LABELS[m] for m in methods], rotation=70, ha='right', fontsize=6.5)
            ax.set_title(title, fontsize=10)
            ax.grid(axis='y', linestyle=':', alpha=.45)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        axes[0].set_ylabel(ylabel)
        fig.tight_layout()
        for extension in ('pdf', 'png'):
            fig.savefig(dest / (basename + '.' + extension), dpi=240, bbox_inches='tight')
        plt.close(fig)
        figures.append(basename)
    return figures


def insertion_text(dest, macro, pairs, suite, tables):
    if not tables:
        raise ValueError('No complete manuscript comparison tables')
    means = {(r['family'], r['scope'], r['metric'], r['method']): r for r in macro if r['suite'] == suite}
    lines = [
        '% Generated only from verified original test-frame scores and locked classical calibration.',
        '\\subsection{Occupied-Band Model-Based Comparators}',
        'We additionally evaluate joint occupied-band OCE-BEM and DPSS-BEM ridge estimators '
        'based on basis-expansion channel models~\\cite{tangleus,zemen}. '
        'Both estimators use the known pilots, the complete occupied received pilot-block band, '
        'and the specified delay/Doppler envelope. Their regularization is selected on an independent '
        'calibration set and fixed before test evaluation. The comparison reuses the original 512 '
        'frames per condition and the existing learned checkpoints.',
    ]
    if suite == 'core':
        lines.append('This comparison covers 11 dynamic channel profiles at 800, 1200, and 1600 Hz, '
                     'with CP length 16 and pilot spacing 4. QPSK NMSE uses SNRs of 15 and 25 dB '
                     '(66 conditions); 16-QAM BER uses 20 dB (33 conditions). '
                     'Four 3GPP profiles are excluded from neural training.')
    lines.append('Pointwise 95\\% bootstrap intervals resample the common eight-frame generation '
                 'clusters and, for learned estimators, the three training seeds.')
    key = ('nmse_qpsk', 'gpp_heldout', 'nmse')
    if all(key + (m,) in means for m in ('dopnet',) + NEW_CLASSICAL):
        a = means[key + ('dopnet',)]['mean_db']
        b = means[key + (NEW_CLASSICAL[0],)]['mean_db']
        c = means[key + (NEW_CLASSICAL[1],)]['mean_db']
        grid_description = 'high-mobility grid' if suite == 'core' else 'matched grid'
        lines.append(f'On the held-out profiles in this {grid_description}, DOP-Net, occupied-band '
                     f'OCE-BEM, and occupied-band DPSS-BEM obtain NMSEs of {a:.3f}, {b:.3f}, '
                     f'and {c:.3f} dB, respectively (Table~\\ref{{tab:classical_nmse}}).')
    key = ('receiver_16qam_20db', 'selected_dynamic', 'ber_diagonal')
    if all(key + (m,) in means for m in ('dopnet',) + NEW_CLASSICAL):
        vals = [100 * means[key + (m,)]['mean'] for m in ('dopnet',) + NEW_CLASSICAL]
        lines.append(f'Across all eleven dynamic channel cases in this matched grid, '
                     f'the common-diagonal channel-balanced BERs are '
                     f'{vals[0]:.3f}\\%, {vals[1]:.3f}\\%, and {vals[2]:.3f}\\%, respectively '
                     '(Table~\\ref{tab:classical_ber}). Estimator-specific ICI-aware detection is '
                     'reported separately: the new classical estimators use full occupied-band ICI '
                     'operators, while the learned estimators retain their five-band interfaces. '
                     'These receiver implementations have different computational costs.')
    lines += [''] + ['\\input{classical_results/' + name + '.tex}' for name in tables
                    if name != 'classical_native_ber'] + ['']
    (dest / 'paper_insertion.tex').write_text('\n\n'.join(lines), encoding='utf-8')
    if 'classical_native_ber' in tables:
        native_lines = [
            '% Supplementary receiver comparison; this is not the common-diagonal-detector control.',
            '\\subsection{Estimator-Specific ICI-Aware Detection}',
            'Table~\\ref{tab:classical_native_ber} reports ICI-aware detection with each estimator\'s '
            'channel interface. The occupied-band BEM estimators use full occupied-band ICI operators. '
            'The learned estimators use their original five-band approximations. This comparison '
            'measures the receiver configurations\' BER.',
            '\\input{classical_results/classical_native_ber.tex}',
            '',
        ]
        (dest / 'native_receiver_insertion.tex').write_text('\n\n'.join(native_lines), encoding='utf-8')


def archive_lock(out, manifest, suite):
    """Verify completed score evidence without requiring raw waveform files.

    This mode checks the server's recorded completion evidence and score hashes.
    It does not repeat the server's raw-observation verification or estimation.
    """
    path = out / 'calibration_lock.json'
    lock = read_json(path)
    if (lock.get('schema_version') != 1 or lock.get('status') != 'locked'
            or lock.get('signature') != object_hash({k: v for k, v in lock.items() if k != 'signature'})
            or lock.get('code_sha256') != code_hashes()
            or lock.get('export_signature') != manifest['export_signature']
            or lock.get('reference_provenance') != manifest['reference_provenance']
            or lock.get('cfg_system_sha256') != object_hash(manifest['cfg_system'])
            or lock.get('geometry_sha256') != object_hash(manifest['geometry'])):
        raise ValueError('Archived calibration lock/code/export binding differs')
    base = {k: v for k, v in manifest.items() if k not in ('export_signature', 'cells', 'suites')}
    if object_hash(base) != manifest['export_signature']:
        raise ValueError('Archived export manifest signature differs')
    # Keep this evidence separate from regenerated report output.
    evidence_path = out / 'server_report' / 'statistics.json'
    evidence = read_json(evidence_path)
    expected = set(manifest['suites'][suite])
    if (evidence.get('paper_ready') is not True or evidence.get('status') != 'complete'
            or evidence.get('suite') != suite
            or evidence.get('expected_cells') != len(expected)
            or evidence.get('completed_verified_cells') != len(expected)
            or evidence.get('missing_cell_ids') != []
            or evidence.get('calibration_lock_sha256') != sha256(path)
            or evidence.get('export_manifest_sha256') != sha256(out / 'export_manifest.json')
            or set(evidence.get('cell_files', {})) != expected):
        raise ValueError('Archived server report is missing or inconsistent')
    for name, expected_hash in evidence.get('report_asset_sha256', {}).items():
        asset = (evidence_path.parent / name).resolve()
        try:
            asset.relative_to(evidence_path.parent.resolve())
        except ValueError:
            raise ValueError('Archived report asset leaves server_report')
        if sha256(asset) != expected_hash:
            raise ValueError('Archived server report asset changed: ' + name)
    for cid, expected_hash in evidence['cell_files'].items():
        if sha256(out / 'cells' / (cid + '.npz')) != expected_hash:
            raise ValueError('Archived server score changed: ' + cid)
    return lock, path


def report(args):
    out = Path(args.classical_output).resolve()
    manifest_path = Path(args.export_manifest).resolve() if args.export_manifest else out / 'export_manifest.json'
    manifest = read_json(manifest_path)
    raw_root = manifest_path.parent
    suite = args.suite
    suites = manifest.get('suites', {})
    if suite not in suites:
        raise ValueError('Requested predeclared suite is missing: ' + suite)
    expected_ids = suites[suite]
    if not isinstance(expected_ids, list) or len(expected_ids) != len(set(expected_ids)):
        raise ValueError('Suite must contain unique predeclared cell IDs')
    metadata = {c['cell_id']: c for c in manifest['cells']}
    if not all(c in metadata for c in expected_ids):
        raise ValueError('Manifest suite contains unknown cell IDs')
    if suite == 'core':
        validate_core([metadata[c] for c in expected_ids])
    frames = int(manifest['samples_per_cell'])
    if frames != 512 or int(manifest['batch_size']) != 8:
        raise ValueError('Matched historical evaluation requires 512 frames in batches of eight')
    lock_path = out / 'calibration_lock.json'
    if not lock_path.is_file():
        raise ValueError('Missing calibration_lock.json: report cannot certify test results')
    archived = getattr(args, 'archived_scores', False)
    if archived:
        lock, checked_lock_path = archive_lock(out, manifest, suite)
    else:
        lock, checked_lock_path = validated_lock(
            argparse.Namespace(data=raw_root, output=out, lock=lock_path), manifest)
    if checked_lock_path != lock_path or lock.get('locked') is not True:
        raise ValueError('Calibration must be explicitly locked before reporting')
    if lock.get('methods') != list(NEW_CLASSICAL):
        raise ValueError('Calibration lock must contain both predeclared classical methods')
    if lock.get('calibration_seed') != 810000001 or lock.get('calibration_samples') != 128:
        raise ValueError('Calibration lock has a different independent-calibration seed/count')
    for method in NEW_CLASSICAL:
        setting = lock['settings'][method]
        if setting.get('method') != method:
            raise ValueError('Calibration settings method differs: ' + method)
        candidate = [c for c in lock['candidates'][method]
                     if c.get('status') == 'complete' and c.get('completed_frames') == 128
                     and c.get('relative_ridge') == setting['relative_ridge']]
        if len(candidate) != 1 or len(candidate[0].get('frame_nmse', [])) != 128:
            raise ValueError('Selected classical ridge lacks all 128 calibration frames')
        values = np.asarray(candidate[0]['frame_nmse'], dtype=float)
        if (not np.isfinite(values).all() or np.any(values < 0)
                or not np.isclose(values.mean(), candidate[0]['mean_linear_nmse'], rtol=1e-12, atol=1e-14)):
            raise ValueError('Invalid calibration-selection scores')
    lock_hash = sha256(lock_path)
    evaluation = read_json(out / 'evaluation.json')
    if (evaluation.get('signature') != object_hash({k: v for k, v in evaluation.items() if k != 'signature'})
            or evaluation.get('methods') != list(NEW_CLASSICAL)
            or evaluation.get('calibration_lock_sha256') != lock_hash
            or evaluation.get('export_signature') != manifest['export_signature']
            or evaluation.get('code_sha256') != code_hashes()
            or evaluation.get('reference_provenance') != manifest['reference_provenance']):
        raise ValueError('CPU evaluation signature, methods, code, or export provenance differs')
    complete_path = out / 'COMPLETE.json'
    complete = read_json(complete_path) if complete_path.is_file() else {}
    cell_bindings = bindings(complete)
    claimed_lock = complete.get('calibration_lock_sha256', complete.get('lock_sha256'))
    claimed_manifest = complete.get('export_manifest_sha256')
    completion_proof = (complete_path.is_file() and claimed_lock == lock_hash
                        and claimed_manifest == sha256(manifest_path)
                        and complete.get('status') == 'complete'
                        and complete.get('signature') == evaluation['signature']
                        and complete.get('methods') == list(NEW_CLASSICAL)
                        and complete.get('export_signature') == manifest['export_signature'])
    dest = Path(args.destination).resolve() if args.destination else out / 'summary'
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    seed_draws = rng.integers(0, 3, size=(args.bootstrap, 3))
    reference = Reference(args.reference_results)
    rows, paired, diagnostic = [], [], []
    buckets = defaultdict(lambda: defaultdict(dict))
    verified = []
    missing = []
    cell_hashes = {}
    try:
        reference.verify_provenance(manifest['reference_provenance'])
        if suite == 'all' and (len(expected_ids) != 636 or set(expected_ids) != set(reference.metadata)):
            raise ValueError('All suite must cover the complete original 636-cell matrix')
        for cell_id in expected_ids:
            file_path = out / 'cells' / (cell_id + '.npz')
            if not file_path.is_file():
                missing.append(cell_id)
                continue
            meta = metadata[cell_id]
            if cell_id not in reference.metadata or meta != reference.metadata[cell_id]:
                raise ValueError('Export metadata differs from original evaluation: ' + cell_id)
            original = reference.cell(cell_id)
            old_hash = str(original['realization_sha256'].item())
            if archived:
                marker = read_json(file_path.with_suffix('.COMPLETE.json'))
                if (marker.get('cell_id') != cell_id or marker.get('samples') != frames
                        or marker.get('methods') != list(NEW_CLASSICAL)
                        or marker.get('realization_sha256') != old_hash):
                    raise ValueError('Archived completion identity differs: ' + cell_id)
                source_hash = marker.get('source_complete_sha256', '')
                if len(source_hash) != 64 or any(c not in '0123456789abcdef' for c in source_hash):
                    raise ValueError('Missing archived raw-export completion binding')
            else:
                raw_complete, raw_complete_path = read_complete(raw_root, manifest, 'test', cell_id)
                if raw_complete['realization_sha256'] != old_hash:
                    raise ValueError('Raw waveform fingerprint differs from original test: ' + cell_id)
                source_hash = sha256(raw_complete_path)
            with np.load(file_path, allow_pickle=False) as z:
                methods = [str(m) for m in z['methods'].tolist()]
                if methods != list(NEW_CLASSICAL):
                    raise ValueError('Expected both predeclared classical methods, in fixed order')
                if scalar(z, 'cell_id') != cell_id or scalar(z, 'realization_sha256') != old_hash:
                    raise ValueError('Classical NPZ belongs to different test frames: ' + cell_id)
                if scalar(z, 'lock_sha256') != lock_hash:
                    raise ValueError('Classical NPZ used a different calibration lock: ' + cell_id)
                if scalar(z, 'source_complete_sha256') != source_hash:
                    raise ValueError('Classical NPZ raw-export binding differs: ' + cell_id)
                if not np.array_equal(z['frame_index'], np.arange(frames)):
                    raise ValueError('Classical NPZ frame ordering differs: ' + cell_id)
                cluster = np.asarray(z['cluster_ids'])
                if not np.array_equal(cluster, original['cluster_id']):
                    raise ValueError('Classical NPZ generation-cluster ordering differs: ' + cell_id)
                if not np.array_equal(cluster, np.arange(frames) // 8):
                    raise ValueError('Expected the original contiguous eight-frame generation clusters')
                classical = {key: npz_matrix(z, key, len(methods), frames).copy()
                             for key in ('nmse', 'errors_diagonal', 'errors_native', 'bits')}
                if scalar(z, 'signature') != evaluation['signature']:
                    raise ValueError('Classical NPZ evaluation signature differs: ' + cell_id)
                if meta['ber'] and not np.all(classical['bits'] == original['bits'][reference.indices['dopnet'][0]]):
                    raise ValueError('Classical and neural BER bit totals differ: ' + cell_id)
                for key in ('fit_relative_residual', 'native_relative_residual',
                            'fit_iterations', 'native_iterations'):
                    values = npz_matrix(z, key, len(methods), frames)
                    if not np.isfinite(values).all() or np.any(values < 0):
                        raise ValueError('Invalid solver diagnostics: ' + key)
                    for m, value in zip(methods, values):
                        settings = lock['settings'][m]
                        budget = settings['fit_max_iter' if key == 'fit_iterations' else 'detector_max_iter'] if key.endswith('iterations') else settings['fit_tolerance' if key == 'fit_relative_residual' else 'detector_tolerance']
                        if np.any(value > budget * (1 + 1e-6) + 1e-12):
                            raise ValueError('Solver diagnostic exceeds locked tolerance/budget: ' + m + '.' + key)
                        diagnostic.append(dict(cell_id=cell_id, method=m, diagnostic=key,
                                               mean=float(value.mean()), max=float(value.max())))
            actual_file_hash = sha256(file_path)
            cell_marker = read_json(file_path.with_suffix('.COMPLETE.json'))
            if (cell_marker.get('status') != 'complete'
                    or cell_marker.get('signature') != evaluation['signature']
                    or cell_marker.get('source_complete_sha256') != source_hash
                    or cell_marker.get('lock_sha256') != lock_hash
                    or cell_marker.get('npz_sha256') != actual_file_hash):
                raise ValueError('Uncommitted or mismatched CPU per-cell completion marker: ' + cell_id)
            cell_hashes[cell_id] = actual_file_hash
            if cell_bindings.get(cell_id) != actual_file_hash:
                completion_proof = False
            # Every method and training seed uses the same cluster resampling weights.
            ids, weights = resampling_weights(cluster, args.bootstrap, rng)
            metrics = ['nmse'] + (['ber_diagonal', 'ber_native'] if meta['ber'] else [])
            for metric in metrics:
                scores = {}
                for m in LEARNED + OLD_CLASSICAL:
                    idx = reference.indices[m]
                    if metric == 'ber_native' and not np.all(original['native_available'][idx]):
                        continue
                    values = original['nmse'][idx] if metric == 'nmse' else original['errors_' + metric[4:]][idx]
                    bits = None if metric == 'nmse' else original['bits'][idx]
                    scores[m] = metric_samples(values, bits, cluster, weights, ids)
                for i, m in enumerate(methods):
                    values = classical['nmse'][i:i + 1] if metric == 'nmse' else classical['errors_' + metric[4:]][i:i + 1]
                    bits = None if metric == 'nmse' else classical['bits'][i:i + 1]
                    scores[m] = metric_samples(values, bits, cluster, weights, ids)
                context = {k: v for k, v in meta.items() if k not in ('studies', 'ber')}
                context.update(suite=suite, frames=frames, test_clusters=len(ids))
                for m, (point, boot) in scores.items():
                    row, _ = interval_row(context, m, metric, point, boot, seed_draws)
                    rows.append(row)
                    if m != 'dopnet':
                        paired.append(paired_row(context, m, metric, point, boot,
                                                 *scores['dopnet'], seed_draws))
                    for family, scope in memberships(meta):
                        if (metric == 'nmse') != (family == 'nmse_qpsk'):
                            continue
                        buckets[(family, scope, metric, m)][meta['channel']][cell_id] = (point, boot)
            verified.append(cell_id)
    finally:
        reference.close()
    all_frames_verified = len(verified) == len(expected_ids)
    paper_ready = all_frames_verified and completion_proof and suite in ('core', 'all')
    macro, macro_paired = [], []
    expected_groups = defaultdict(set)
    for cell_id in expected_ids:
        for family, scope in memberships(metadata[cell_id]):
            expected_groups[(family, scope)].add(cell_id)
    macro_scores = {}
    for (family, scope, metric, method), channels in buckets.items():
        available_ids = {cid for data in channels.values() for cid in data}
        if available_ids != expected_groups[(family, scope)]:
            continue
        channel_points = [np.mean([v[0] for v in data.values()], axis=0) for data in channels.values()]
        channel_boots = [np.mean([v[1] for v in data.values()], axis=0) for data in channels.values()]
        point = np.mean(channel_points, axis=0)
        boot = np.mean(channel_boots, axis=0)
        context = dict(suite=suite, family=family, scope=scope, channel_count=len(channels),
                       cells=len(available_ids), frames=frames * len(available_ids),
                       aggregation='equal cells within channel, then equal channels',
                       complete_predeclared_group=True)
        row, _ = interval_row(context, method, metric, point, boot, seed_draws)
        macro.append(row)
        macro_scores[(family, scope, metric, method)] = (point, boot, context)
    for (family, scope, metric, method), (point, boot, context) in macro_scores.items():
        if method == 'dopnet':
            continue
        ref = macro_scores.get((family, scope, metric, 'dopnet'))
        if ref is not None:
            macro_paired.append(paired_row(context, method, metric, point, boot, ref[0], ref[1], seed_draws))
    write_csv(dest / 'cell_summary.csv', rows)
    write_csv(dest / 'cell_paired_differences.csv', paired)
    write_csv(dest / 'macro_summary.csv', macro)
    write_csv(dest / 'macro_paired_differences.csv', macro_paired)
    write_csv(dest / 'solver_diagnostics.csv', diagnostic)
    statistics = dict(schema_version=1, suite=suite, expected_cells=len(expected_ids),
                      completed_verified_cells=len(verified), missing_cell_ids=missing,
                      paper_ready=paper_ready, bootstrap_repetitions=args.bootstrap,
                      bootstrap_seed=args.bootstrap_seed, calibration_lock_sha256=lock_hash,
                      export_manifest_sha256=sha256(manifest_path), cell_files=cell_hashes,
                      learned_training_seeds=list(SEEDS), classical_training_seeds=[],
                      aggregation='frame NMSE in linear units; per-cell BER total errors/total bits; '
                                  'equal cells within channel, equal channels',
                      intervals=dict(learned=CI_LEARNED, classical=CI_CLASSICAL),
                      paired='shared frame-cluster resampling, shared seed resampling across learned estimators',
                      status='complete' if paper_ready else 'partial or not fully attested',
                      manuscript_fragments_generated=paper_ready)
    statistics['verification_mode'] = ('completed server score archive; raw observations not reread'
                                       if archived else 'raw observations and completed scores')
    statistics['reporter_sha256'] = sha256(Path(__file__))
    if suite == 'core':
        statistics['grid'] = '11 dynamic channels; Doppler 800/1200/1600 Hz; QPSK SNR 15/25 dB '
        statistics['grid'] += 'and 16-QAM SNR 20 dB; CP 16; pilot spacing 4; delay spread 300 ns'
    if paper_ready:
        tables = write_tables(dest, macro, suite)
        statistics['figures'] = make_plots(dest, macro, suite)
        insertion_text(dest, macro, macro_paired, suite, tables)
    else:
        # A stale manuscript fragment from an earlier run must never be offered as current evidence.
        stale = dest / 'paper_insertion.tex'
        if stale.exists():
            archived = dest / ('paper_insertion.stale.' + sha256(stale)[:12] + '.tex')
            if archived.exists() and sha256(archived) == sha256(stale):
                stale.unlink()
            else:
                stale.rename(archived)
    assets = ['cell_summary.csv', 'cell_paired_differences.csv', 'macro_summary.csv',
              'macro_paired_differences.csv', 'solver_diagnostics.csv']
    if paper_ready:
        assets += [name + '.tex' for name in tables]
        assets += ['paper_insertion.tex']
        if 'classical_native_ber' in tables:
            assets += ['native_receiver_insertion.tex']
        assets += [name + '.' + extension for name in statistics['figures']
                   for extension in ('pdf', 'png')]
    statistics['report_asset_sha256'] = {name: sha256(dest / name) for name in assets}
    write_json(dest / 'statistics.json', statistics)
    (dest / 'README.md').write_text(
        '# Verified classical comparison\n\n'
        f'Suite: `{suite}`. Verified cells: {len(verified)}/{len(expected_ids)}. '
        f'Paper insertion ready: {paper_ready}.\n\n'
        'The core suite is a targeted high-mobility comparison, not the entire 636-condition matrix. '
        'It contains 66 QPSK NMSE conditions and 33 16-QAM receiver conditions. Each uses the '
        'original 512 frames, grouped into 64 independent eight-frame generation batches.\n\n'
        'NMSE is averaged as a linear frame error before conversion to dB. BER is computed from '
        'the total bit-error and bit counts within each cell. Cells are averaged within channel, '
        'then channels receive equal weight.\n\n'
        'The 95% percentile intervals resample common frame-generation clusters for all methods. '
        'For learned estimators, the three training seeds are additionally resampled globally, '
        'with the same seed draws for all learned methods. Classical estimators have no training seeds. '
        'These joint intervals differ from the existing paper tables\' seed-t intervals; they must '
        'keep their own caption and must not silently overwrite those earlier columns.\n\n'
        'The CSV files describe only actually verified score files. Macro rows require a complete '
        'predeclared group. Manuscript tables and text require the whole declared core/all suite, '
        'the raw-export waveform fingerprints, calibration lock, and score-file completion hashes '
        'to agree. A custom ad hoc cell subset is never automatically promoted to manuscript evidence.\n\n'
        'Use `paper_insertion.tex` together with the three table fragments after reviewing the actual '
        'results. Its main-text insertion includes NMSE and common-diagonal BER. '
        '`native_receiver_insertion.tex` is a separate supplementary comparison of estimator-specific '
        'ICI receivers: new classical models use full occupied-band operators and learned models use '
        'their original five-band interfaces. These computational costs are not equated. '
        'The script does not change the main manuscript or assert that DOP-Net wins.\n',
        encoding='utf-8')
    print(json.dumps({k: statistics[k] for k in ('suite', 'expected_cells', 'completed_verified_cells', 'paper_ready')}, indent=2))
    return paper_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classical-output', required=True, help='Directory containing CPU per-cell score NPZ files')
    parser.add_argument('--reference-results', required=True, help='Original learned/classical results folder or ZIP')
    parser.add_argument('--export-manifest', help='Raw export export_manifest.json; required unless copied to output root')
    parser.add_argument('--suite', default='core', help='Predeclared export suite, normally core')
    parser.add_argument('--destination', help='Defaults to CLASSICAL_OUTPUT/summary')
    parser.add_argument('--archived-scores', action='store_true',
                        help='Rebuild from completed scores and preserved server_report; no raw replay/inference')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=920260930)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        parser.error('Use at least 1000 bootstrap repetitions')
    ready = report(args)
    # Partial diagnostic summaries are useful, but cannot be mistaken for submission evidence.
    return 0 if ready else 2


if __name__ == '__main__':
    sys.exit(main())
