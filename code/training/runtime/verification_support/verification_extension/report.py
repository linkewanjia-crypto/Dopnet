"""Score-only corrected classical comparisons and digital receiver tables.

No waveform replay, inference or calibration occurs here. Historical learned
scores are reused; corrected baselines have new method IDs and independently
locked calibration. Only a complete declared suite produces manuscript assets.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

from .protocol import METHODS, digest, file_hash, read_json, select_suite, write_json


def _legacy_helpers():
    # Reuse the established score reader and mathematical bootstrap helpers,
    # without invoking its obsolete raw-export/calibration execution contract.
    directory = Path(__file__).resolve().parents[1] / 'classical_extension'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('_dopnet_saved_score_math', directory / 'report_classical.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_old = _legacy_helpers()
SEEDS = (42, 43, 44)
LEARNED = ('dopnet', 'dopnet_lite', 'slow_time', 'fortitran', 'cfr_derived', 'pilot_only_fast_time')
OLD_CLASSICAL = ('ls_interp', 'lmmse_wiener', 'dpsbem_ls')
ORDER = OLD_CLASSICAL + tuple(METHODS) + LEARNED
LABELS = {**_old.LABELS, 'cfr_derived': 'CFR-derived control',
          'pilot_only_fast_time': 'Pilot-support fast-time control',
          'pilot_covariance_lmmse': 'Pilot covariance LMMSE',
          'oce_bem_joint_cholesky': 'Joint OCE-BEM (recalibrated)',
          'dpss_bem_joint_cholesky': 'Joint DPSS-BEM (recalibrated)'}
SEEN, HELDOUT = _old.SEEN, _old.HELDOUT
STATIC = ('awgn', 'rayleigh', 'rician', 'nakagami')
CORE_CHANNELS = ('doubly_selective',) + SEEN + HELDOUT
CI_CLASSICAL = 'paired generation-cluster percentile bootstrap'
CI_LEARNED = 'joint training-seed and paired generation-cluster percentile bootstrap'


class Reference(_old.Reference):
    def __init__(self, path):
        path = Path(path).resolve()
        if path.name == 'cells' and (path.parent / 'evaluation.json').is_file():
            path = path.parent
        super().__init__(path)
        if len(self.metadata) != 636:
            raise ValueError('The frozen reference must retain all 636 original score cells')
        for method in LEARNED + OLD_CLASSICAL:
            idx = [i for i, r in enumerate(self.routes) if r['method'] == method]
            expected = list(SEEDS) if method in LEARNED else [None]
            if [self.routes[i]['training_seed'] for i in idx] != expected:
                raise ValueError('Missing/reordered reference routes for ' + method)
            self.indices[method] = idx

    def cell_hash(self, cid):
        if self.archive is not None:
            return hashlib.sha256(self.archive.read(self.root + 'cells/' + cid + '.npz')).hexdigest()
        return file_hash(self.root / 'cells' / (cid + '.npz'))


def signed(path):
    value = read_json(path)
    if value.get('signature') != digest({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Incorrect signed metadata: ' + str(path))
    return value


def validate_calibration_lock(lock, output):
    """All BEM candidates must be fully solved, not numerical losers."""
    count = lock.get('calibration_samples', lock.get('calibration_count', lock.get('calibration_frames')))
    if lock.get('calibration_seed') != 810000001 or count != 128:
        raise ValueError('BEM references require the declared independent 128-frame calibration ensemble')
    for method in METHODS[1:]:
        candidates = lock.get('candidates', {}).get(method, [])
        if not candidates:
            raise ValueError('Missing complete BEM calibration grid: ' + method)
        selected = lock.get('settings', {}).get(method)
        chosen = []
        for candidate in candidates:
            values = np.asarray(candidate.get('frame_nmse', []), dtype=np.float64)
            if (candidate.get('status') != 'complete' or candidate.get('completed_frames') != 128
                or values.shape != (128,) or not np.isfinite(values).all() or np.any(values < 0)
                or not math.isclose(float(values.mean()), float(candidate.get('mean_linear_nmse', -1)), rel_tol=1e-10, abs_tol=1e-13)):
                raise ValueError('Every declared BEM candidate must complete the same 128-frame calibration: ' + method)
            if not math.isfinite(float(candidate.get('solver_residual_max', math.inf))) or float(candidate['solver_residual_max']) > 1e-6:
                raise ValueError('Every BEM calibration candidate needs its accepted true residual: ' + method)
            if candidate.get('settings') == selected:
                chosen.append(candidate)
        if len(chosen) != 1:
            raise ValueError('Selected BEM setting is not bound to one complete calibration candidate: ' + method)
        best = min(float(c['mean_linear_nmse']) for c in candidates)
        if float(chosen[0]['mean_linear_nmse']) > best + 1e-12:
            raise ValueError('Locked BEM dictionary/ridge does not minimize declared calibration NMSE: ' + method)
        grid = lock.get('contract', {}).get('grid')
        if grid is not None:
            expected_settings = {(int(q), int(window[0]), int(window[1]), float(rho))
                                 for q in grid['q_grid'] for window in grid['delay_windows'] for rho in grid['ridge_grid']}
            actual_settings = {(int(c['settings']['q']), int(c['settings']['delay_start']),
                                int(c['settings']['delay_stop']), float(c['settings']['relative_ridge'])) for c in candidates}
            if len(candidates) != len(expected_settings) or actual_settings != expected_settings:
                raise ValueError('Declared BEM dictionary/ridge grid is incomplete or duplicated: ' + method)
    priors = lock.get('statistical_priors', {})
    if not priors:
        raise ValueError('Missing independently calibrated statistical covariance priors')
    total = 0
    root = Path(output).resolve()
    for key, prior in priors.items():
        fit, held, all_frames = (int(prior[name]) for name in ('fit_frames', 'holdout_frames', 'all_frames'))
        if fit < 16 or held < 8 or fit + held != all_frames:
            raise ValueError('Statistical calibration must preserve adequate disjoint whole-frame chunks: ' + key)
        total += all_frames
        path = (root / prior['npz_relative_path']).resolve()
        path.relative_to(root)
        if not path.is_file() or file_hash(path) != prior['sha256']:
            raise ValueError('Locked statistical covariance state changed: ' + key)
        if float(prior['noise_scale']) <= 0 or not math.isfinite(float(prior['noise_scale'])):
            raise ValueError('Invalid selected statistical noise regularization')
        noise_candidates = prior.get('noise_candidates', [])
        if not noise_candidates:
            raise ValueError('Missing statistical regularization holdout comparison: ' + key)
        selected_candidates = []
        for candidate in noise_candidates:
            values = np.asarray(candidate.get('frame_nmse', []), dtype=float)
            if (candidate.get('status') != 'complete' or candidate.get('frames') != held or values.shape != (held,)
                or not np.isfinite(values).all() or np.any(values < 0)
                or not math.isclose(float(values.mean()), float(candidate.get('mean_linear_nmse', -1)), rel_tol=1e-10, abs_tol=1e-13)
                or not math.isfinite(float(candidate.get('solver_residual_max', math.inf)))
                or float(candidate.get('solver_residual_max', math.inf)) > 1e-6):
                raise ValueError('Every statistical noise candidate needs its complete independent holdout scores: ' + key)
            if candidate.get('noise_scale') == prior['noise_scale']:
                selected_candidates.append(candidate)
        if len(selected_candidates) != 1 or float(selected_candidates[0]['mean_linear_nmse']) > min(
            float(c['mean_linear_nmse']) for c in noise_candidates) + 1e-12:
            raise ValueError('Statistical noise scale was not chosen from the full held-out calibration: ' + key)
    statistical_count = int(lock.get('statistical_calibration_samples', 128))
    omitted = sum(int(v['frames']) for v in lock.get('unavailable_statistical_priors', {}).values())
    if total + omitted != statistical_count:
        raise ValueError('Statistical prior groups must partition their entire declared independent ensemble')


def validate_required_priors(lock, metadata):
    """A skipped optional calibration CP must never become an evaluated prior."""
    valid_cps = set()
    for prior in lock['statistical_priors'].values():
        geometry = prior['geometry']
        active_cp = float(geometry['cp_wave_samples']) * int(geometry['n_active']) / int(geometry['nfft'])
        if not math.isclose(active_cp, round(active_cp), abs_tol=1e-10):
            raise ValueError('Locked covariance has a noninteger active-grid CP')
        valid_cps.add(int(round(active_cp)))
    required = {int(meta['cp_len']) for meta in metadata}
    if not required <= valid_cps:
        raise ValueError('The requested test suite needs independently calibrated priors for CP values: ' + str(sorted(required - valid_cps)))


def scalar(arrays, key):
    return _old.scalar(arrays, key)


def interval_row(context, method, metric, point, boot, seed_draws):
    mean = float(np.mean(point))
    draws = _old.final_draws(boot, seed_draws)
    lo, hi = np.quantile(draws, [.025, .975])
    row = dict(context, method=method, estimator=LABELS[method], metric=metric,
               n_training_seeds=3 if method in LEARNED else 0,
               mean=mean, ci95_lo=float(lo), ci95_hi=float(hi),
               interval=CI_LEARNED if method in LEARNED else CI_CLASSICAL)
    if metric == 'nmse':
        if mean <= 0 or lo <= 0:
            raise ValueError('Nonpositive NMSE cannot support a finite dB table')
        row.update(mean_db=float(10 * np.log10(mean)), ci95_lo_db=float(10 * np.log10(lo)),
                   ci95_hi_db=float(10 * np.log10(hi)))
    else:
        row['zero_error_observation'] = bool(mean == 0)
    if method in LEARNED:
        row.update({f'seed{s}': float(v) for s, v in zip(SEEDS, point)})
    return row


def paired_row(context, method, metric, point, boot, ref_point, ref_boot, seed_draws):
    return _old.paired_row(context, method, metric, point, boot, ref_point, ref_boot, seed_draws)


def memberships(meta):
    """Fixed groups derived from the full frozen protocol, never available cells."""
    nominal = _old.is_nominal(meta)
    scopes = ['all']
    scopes.append('static' if meta['channel'] in STATIC else 'dynamic_all')
    if meta['channel'] == 'doubly_selective':
        scopes.append('dynamic_synthetic')
    if meta['channel'] in SEEN + HELDOUT:
        scopes += ['gpp_all', 'gpp_seen' if meta['channel'] in SEEN else 'gpp_heldout']
        if float(meta['doppler_hz']) >= 800:
            scopes.append('gpp_high_mobility')
    out = []
    if nominal and meta['modulation'] == 'qpsk' and 'nmse' in meta['studies']:
        out += [('nominal_nmse', scope, 'nmse') for scope in scopes]
    is_core = (nominal and meta['channel'] in CORE_CHANNELS and float(meta['doppler_hz']) in (800., 1200., 1600.)
               and ((meta['modulation'] == 'qpsk' and float(meta['snr_db']) in (15., 25.) and 'nmse' in meta['studies'])
                    or (meta['modulation'] == '16qam' and float(meta['snr_db']) == 20. and 'receiver' in meta['studies'])))
    if is_core:
        core_scopes = ['selected_dynamic', 'dynamic_fd_%g' % float(meta['doppler_hz'])]
        if meta['channel'] in SEEN:
            core_scopes += ['gpp_seen', 'seen_fd_%g' % float(meta['doppler_hz'])]
        if meta['channel'] in HELDOUT:
            core_scopes += ['gpp_heldout', 'heldout_fd_%g' % float(meta['doppler_hz'])]
        family = 'nmse_qpsk' if meta['modulation'] == 'qpsk' else 'receiver_16qam_20db'
        metrics = ('nmse',) if family == 'nmse_qpsk' else ('ber_diagonal', 'ber_native')
        out += [(family, scope, metric) for scope in core_scopes for metric in metrics]
    # Broader robustness groups are offered only when the entire original
    # membership set is scored, rather than relabeling the core subset as all.
    for study in meta['studies']:
        family = 'study_' + study + '_' + meta['modulation']
        metrics = ('nmse', 'ber_diagonal', 'ber_native') if meta['ber'] else ('nmse',)
        out += [(family, scope, metric) for scope in scopes for metric in metrics]
    return list(dict.fromkeys(out))


def aggregate_complete_groups(buckets, expected):
    """Equal conditions within channel, then equal channel weighting.

Point arrays preserve the three seed columns until the final mean. Bootstrap
arrays preserve a shared draw index and seed columns, so paired differences
use the same realization/seed weights even after macro aggregation.
"""
    result = {}
    for (family, scope, metric, method), channels in buckets.items():
        available = {cid for group in channels.values() for cid in group}
        if available != expected[(family, scope, metric)]:
            continue
        points = [np.mean([x[0] for x in group.values()], axis=0) for group in channels.values()]
        boots = [np.mean([x[1] for x in group.values()], axis=0) for group in channels.values()]
        result[(family, scope, metric, method)] = (
            np.mean(points, axis=0), np.mean(boots, axis=0),
            {'family': family, 'scope': scope, 'metric': metric, 'channel_count': len(channels),
             'cells': len(available), 'frames': 512 * len(available),
             'aggregation': 'equal conditions within channel, then equal channels',
             'complete_predeclared_group': True})
    return result


def verify_corrected_cell(output, cid, meta, original, reference_hash, evaluation, lock_hash, root_bindings):
    """Validate identities, score denominators, solver limits and both commits."""
    path = Path(output) / 'cells' / (cid + '.npz')
    marker_path = path.with_suffix('.COMPLETE.json')
    if not path.is_file() or not marker_path.is_file():
        raise ValueError('Missing completed corrected cell: ' + cid)
    actual_sha = file_hash(path)
    marker = read_json(marker_path)
    original_wave = str(original['realization_sha256'].item())
    for key, expected in (('status', 'complete'), ('signature', evaluation['signature']),
                          ('lock_sha256', lock_hash), ('source_score_sha256', reference_hash),
                          ('realization_sha256', original_wave), ('npz_sha256', actual_sha)):
        if marker.get(key) != expected:
            raise ValueError(f'Corrected cell marker mismatch {cid}.{key}')
    if marker.get('meta') != meta:
        raise ValueError('Corrected completed-cell metadata differs: ' + cid)
    if root_bindings.get(cid) != actual_sha:
        raise ValueError('Root completion does not bind this corrected NPZ: ' + cid)
    with np.load(path, allow_pickle=False) as z:
        for key, expected in (('signature', evaluation['signature']), ('lock_sha256', lock_hash),
                              ('source_score_sha256', reference_hash), ('realization_sha256', original_wave)):
            if scalar(z, key) != expected:
                raise ValueError(f'Corrected NPZ binding mismatch {cid}.{key}')
        if list(z['methods'].astype(str)) != list(METHODS):
            raise ValueError('Corrected method order differs')
        if not np.array_equal(z['frame_index'], np.arange(512)):
            raise ValueError('Original/corrected frame order differs')
        # Historical score NPZs did not carry frame_index. Their contiguous
        # cluster_id order plus the original-score SHA binds that ordering.
        if 'frame_index' in original and not np.array_equal(original['frame_index'], np.arange(512)):
            raise ValueError('Original frame ordering differs')
        cluster = np.asarray(z['cluster_ids'])
        if not np.array_equal(cluster, np.arange(512) // 8) or not np.array_equal(cluster, original['cluster_id']):
            raise ValueError('Original/corrected generated-frame clusters differ')
        scores = {key: _old.npz_matrix(z, key, len(METHODS), 512).copy()
                  for key in ('nmse', 'errors_diagonal', 'errors_native', 'bits')}
        if not np.isfinite(scores['nmse']).all() or np.any(scores['nmse'] < 0):
            raise ValueError('Invalid corrected NMSE')
        native = np.asarray(z['native_available'], dtype=bool)
        if native.shape != (len(METHODS),) or not np.array_equal(native, [False, True, True]):
            raise ValueError('Statistical prior has no ICI interface; BEM methods must provide theirs')
        if meta['ber']:
            dopnet_idx = next(i for i, route in enumerate(evaluation['_reference_routes']) if route['method'] == 'dopnet')
            if not np.all(scores['bits'] == original['bits'][dopnet_idx]):
                raise ValueError('Corrected/reference BER denominators differ')
            for key in ('errors_diagonal', 'errors_native'):
                subset = scores[key] if key == 'errors_diagonal' else scores[key][native]
                denominator = scores['bits'] if key == 'errors_diagonal' else scores['bits'][native]
                if (not np.isfinite(subset).all() or not np.equal(subset, np.floor(subset)).all()
                    or np.any(subset < 0) or np.any(subset > denominator) or np.any(denominator <= 0)):
                    raise ValueError('Invalid bit-error counts: ' + key)
            if not np.all(scores['errors_native'][~native] == -1):
                raise ValueError('Unavailable native BER must use sentinel -1, never an artificial zero')
        diagnostics = []
        for key in ('fit_relative_residual', 'native_relative_residual', 'fit_iterations', 'native_iterations'):
            values = _old.npz_matrix(z, key, len(METHODS), 512)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Invalid solver diagnostics: ' + key)
            for mi, method in enumerate(METHODS):
                if key.startswith('native') and (not native[mi] or not meta['ber']):
                    continue
                v = values[mi]
                threshold = float(evaluation.get('solver_residual_tolerance', 1e-6))
                if 'residual' in key and np.any(v > threshold * (1 + 1e-6) + 1e-12):
                    raise ValueError('Solver true residual exceeds declared threshold: ' + method)
                diagnostics.append({'cell_id': cid, 'method': method, 'diagnostic': key,
                                    'mean': float(v.mean()), 'max': float(v.max()),
                                    'accepted_threshold': threshold if 'residual' in key else ''})
    return scores, native, cluster, diagnostics, actual_sha


def _value(row):
    return _old.format_value(row)


def write_tables(dest, macro):
    lookup = {(r['family'], r['scope'], r['metric'], r['method']): r for r in macro}
    specs = [
        ('nominal_nmse', 'nmse', 'corrected_nominal_nmse', 'Nominal-grid NMSE (dB)',
         ('all', 'gpp_seen', 'gpp_heldout'), ('All nominal channels', 'Seen 3GPP', 'Held-out 3GPP')),
        ('nmse_qpsk', 'nmse', 'corrected_core_nmse', 'Matched high-mobility QPSK NMSE (dB)',
         ('selected_dynamic', 'gpp_seen', 'gpp_heldout'), ('Selected dynamic', 'Seen 3GPP', 'Held-out 3GPP')),
        ('receiver_16qam_20db', 'ber_diagonal', 'corrected_core_common_ber', 'Matched high-mobility 16-QAM BER at 20 dB (\\%) with the common diagonal detector',
         ('selected_dynamic', 'gpp_seen', 'gpp_heldout'), ('Selected dynamic', 'Seen 3GPP', 'Held-out 3GPP')),
        ('receiver_16qam_20db', 'ber_native', 'corrected_core_native_ber', 'Estimator-specific ICI receiver BER (\\%): full occupied-band BEM or the learned five-band interface',
         ('selected_dynamic', 'gpp_seen', 'gpp_heldout'), ('Selected dynamic', 'Seen 3GPP', 'Held-out 3GPP'))]
    note = ('Means [pointwise 95\\% intervals] use common eight-frame generation-cluster bootstrap '
            'and additionally the three trained seeds for learned methods. '
            'Linear NMSE or bit-count BER is averaged within each channel, then equally across channels.')
    written = []
    for family, metric, name, caption, scopes, titles in specs:
        available = [m for m in ORDER if any((family, s, metric, m) in lookup for s in scopes)]
        if not available:
            continue
        lines = ['\\begin{table*}[t]', '\\centering', '\\scriptsize', '\\caption{' + caption + '. ' + note + '}',
                 '\\label{tab:' + name + '}', '\\begin{tabular}{lccc}', '\\toprule',
                 'Estimator & ' + ' & '.join(titles) + ' \\\\', '\\midrule']
        for method in available:
            lines.append(LABELS[method] + ' & ' + ' & '.join(_value(lookup.get((family, s, metric, method))) for s in scopes) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
        (dest / (name + '.tex')).write_text('\n'.join(lines), encoding='utf-8')
        written.append(name + '.tex')
    return written


def make_doppler_curves(dest, macro):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = []
    for family, metric, name, ylabel in (
        ('nmse_qpsk', 'nmse', 'corrected_doppler_nmse', 'NMSE (dB)'),
        ('receiver_16qam_20db', 'ber_diagonal', 'corrected_doppler_common_ber', 'Common-detector BER (%)'),
        ('receiver_16qam_20db', 'ber_native', 'corrected_doppler_native_ber', 'Estimator-specific ICI BER (%)')):
        rows = [r for r in macro if r['family'] == family and r['metric'] == metric]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharey=True)
        for ax, prefix, title in zip(axes, ('dynamic_fd_', 'seen_fd_', 'heldout_fd_'),
                                   ('Selected dynamic', 'Seen 3GPP', 'Held-out 3GPP')):
            for method in tuple(METHODS) + ('dopnet', 'dopnet_lite', 'slow_time', 'fortitran'):
                values = sorted([r for r in rows if r['method'] == method and r['scope'].startswith(prefix)],
                                key=lambda r: float(r['scope'][len(prefix):]))
                if len(values) != 3:
                    continue
                x = np.asarray([float(r['scope'][len(prefix):]) for r in values])
                keys, factor = (('mean_db', 'ci95_lo_db', 'ci95_hi_db'), 1.) if metric == 'nmse' else (
                    ('mean', 'ci95_lo', 'ci95_hi'), 100.)
                mean, lo, hi = (np.asarray([r[k] for r in values]) * factor for k in keys)
                line, = ax.plot(x, mean, marker='o', markersize=3, linewidth=1.1, label=LABELS[method])
                ax.fill_between(x, lo, hi, color=line.get_color(), alpha=.13, linewidth=0)
            ax.set_xlabel('Doppler frequency (Hz)')
            ax.set_title(title, fontsize=10)
            ax.set_xticks([800, 1200, 1600])
            ax.grid(alpha=.25, linestyle=':')
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        if not handles:
            plt.close(fig)
            continue
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.03), ncol=3, fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, .85))
        for ext in ('pdf', 'png'):
            filename = name + '.' + ext
            fig.savefig(dest / filename, dpi=250, bbox_inches='tight')
            files.append(filename)
        plt.close(fig)
    return files


def qualified_workload_threads(rows, protocol, recommendations):
    """Select one complete recommended device/thread variant per workload.

Thread alternatives remain separate measurements. A device value is omitted
unless all declared model seeds qualify for the very same thread count.
"""
    host = [r for r in rows if r['stage'] == 'host_origin_total']
    grouped = defaultdict(lambda: defaultdict(list))
    required_by_workload = defaultdict(set)
    for row in host:
        workload = tuple(row[k] for k in ('source_name', 'input_boundary', 'method', 'receiver_mode', 'batch_size'))
        required_by_workload[workload].add(row['seed'])
        grouped[workload][(row['device'], row['cpu_threads'])].append(row)
    selected = defaultdict(dict)
    for workload, variants in grouped.items():
        source, boundary, method, mode, batch = workload
        declared = set(protocol.get('seeds', required_by_workload[workload]))
        eligible = {variant: values for variant, values in variants.items()
                    if len(values) == len(declared) and set(v['seed'] for v in values) == declared}
        recommendation = next((r for r in recommendations if r['method'] == method and r['batch_size'] == batch
                               and r['receiver_mode'] == mode and r['source_name'] == source), {})
        recommended_threads = recommendation.get('best_cpu_threads_by_device', {})
        for device in dict.fromkeys(variant[0] for variant in eligible):
            options = {variant: values for variant, values in eligible.items() if variant[0] == device}
            if device in recommended_threads:
                chosen = (device, int(recommended_threads[device]))
                if chosen not in options:
                    # A stale/partial recommendation cannot produce a table value.
                    continue
            else:
                chosen = min(options, key=lambda variant: float(np.median(
                    [v['latency_ms_p50'] for v in options[variant]])))
            selected[workload][device] = options[chosen]
    return selected


def latency_assets(dest, latency_output, reference_signature):
    """Report only measured, equivalence-qualified digital chain timings."""
    folder = Path(latency_output)
    path = folder if folder.is_file() else folder / 'digital_receiver_latency.json'
    if not path.is_file():
        return [], {'status': 'not_available'}
    timing = read_json(path)
    if not timing.get('complete'):
        return [], {'status': 'incomplete', 'latency_json_sha256': file_hash(path)}
    protocol = timing.get('protocol', {})
    if protocol.get('experiment_signature') != reference_signature:
        raise ValueError('Digital latency and frozen score experiment differ')
    rows = []
    for value in timing.get('rows', []):
        if value.get('status') != 'measured' or not value.get('numerical_equivalence', {}).get('passed'):
            continue
        for stage, summary in value['timing'].items():
            row = {key: value[key] for key in ('method', 'seed', 'batch_size', 'receiver_mode', 'device', 'source_name', 'input_boundary')}
            row['cpu_threads'] = int(value.get('cpu_threads', protocol.get('cpu_threads', 1)))
            row.update(stage=stage, **{k: v for k, v in summary.items() if isinstance(v, (float, int, str)) or v is None})
            rows.append(row)
    _old.write_csv(dest / 'corrected_digital_receiver_latency.csv', rows)
    recommendations = timing.get('device_recommendations', [])
    _old.write_csv(dest / 'corrected_inference_device_choices.csv', recommendations)
    qualified_workloads = qualified_workload_threads(rows, protocol, recommendations)
    def tex(text):
        return str(text).replace('_', '\\_').replace('%', '\\%')
    published = []
    for single, name, caption, heading in (
        (True, 'corrected_digital_receiver_latency',
         'Single-frame digital receiver wall-clock latency (ms). Entries are median / 95th percentile, each averaged over model seeds. '
         'Directly timed host-origin totals include necessary pilot preprocessing, estimation, detection, demapping and device transfers. ',
         'Method & Receiver & CPU median/P95 & CUDA median/P95'),
        (False, 'corrected_digital_receiver_throughput',
         'Batched digital receiver throughput (frames/s). Each entry averages measured host-origin throughput over model seeds. '
         'Frame repeats used for timing are workloads, not additional test samples. ',
         'Method & Receiver & Batch & CPU & CUDA')):
        by_workload = defaultdict(dict)
        for workload, devices in qualified_workloads.items():
            source, boundary, method, mode, batch = workload
            if (batch == 1) != single:
                continue
            by_workload[workload] = devices
        if not by_workload:
            continue
        width, spec = (4, 'llll') if single else (5, 'llrrr')
        lines = ['\\begin{table*}[t]', '\\centering', '\\scriptsize',
                 '\\caption{' + caption + 'Source and input boundary are shown in each block. '
                 'Each device uses its recommended thread setting, denoted by $t$, with all declared model seeds. '
                 'CUDA entries require numerical equivalence to CPU.}',
                 '\\label{tab:' + ('corrected_digital_latency' if single else 'corrected_digital_throughput') + '}',
                 '\\begin{tabular}{' + spec + '}', '\\toprule', heading + ' \\\\', '\\midrule']
        current_source = None
        for workload, devices in sorted(by_workload.items()):
            source, boundary, method, mode, batch = workload
            if (source, boundary) != current_source:
                boundary_label = 'actual received time samples' if boundary == 'actual_received_time_samples' else 'received occupied OFDM grid'
                lines.append('\\multicolumn{' + str(width) + '}{l}{' + tex(source) + ': ' + boundary_label + '} \\\\')
                current_source = (source, boundary)
            cpu = [v for device, group in devices.items() if device == 'cpu' for v in group]
            gpu = [v for device, group in devices.items() if device.startswith('cuda') for v in group]
            def value(values):
                if not values:
                    return '--'
                if single:
                    p50 = np.mean([v['latency_ms_p50'] for v in values])
                    p95 = np.mean([v['latency_ms_p95'] for v in values])
                    return f'{p50:.3f} / {p95:.3f} ($t={values[0]["cpu_threads"]}$)'
                throughput = np.mean([v.get('frames_per_second',
                                            1000 * batch / v['latency_ms_mean']) for v in values])
                return f'{throughput:.1f} ($t={values[0]["cpu_threads"]}$)'
            entries = [LABELS.get(method, method), 'Diagonal' if mode == 'common_diagonal' else 'Own ICI']
            if not single:
                entries.append(str(batch))
            entries += [value(cpu), value(gpu)]
            lines.append(' & '.join(entries) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
        filename = name + '.tex'
        (dest / filename).write_text('\n'.join(lines), encoding='utf-8')
        published.append(filename)
    write_json(dest / 'corrected_latency_scope.json', {'protocol': protocol, 'hardware': timing.get('hardware'),
               'device_recommendations': recommendations, 'latency_json_sha256': file_hash(path),
               'qualified_measured_rows': len({tuple(r[k] for k in ('method', 'seed', 'batch_size', 'receiver_mode', 'device', 'source_name')) for r in rows})})
    return ['corrected_digital_receiver_latency.csv', 'corrected_inference_device_choices.csv',
            'corrected_latency_scope.json'] + published, {'status': 'complete', 'latency_json_sha256': file_hash(path)}


def classical_backend_assets(dest, output, lock):
    """Keep verification cost distinct from deployment/digital-chain timing."""
    path = Path(output) / 'classical_backend_benchmark.json'
    policy_path = Path(output) / 'classical_device_policy.json'
    if not path.is_file() or not policy_path.is_file():
        return [], {'status': 'not_available'}
    benchmark, policy = read_json(path), signed(policy_path)
    if (not benchmark.get('complete') or not policy.get('complete') or policy.get('precision') != 'complex128'
        or policy.get('benchmark_sha256') != file_hash(path)):
        return [], {'status': 'incomplete_or_unbound'}
    if benchmark.get('settings') != lock['settings'] or not benchmark.get('protocol', {}).get('state_directory'):
        return [], {'status': 'preliminary_settings_only'}
    if (benchmark.get('calibration', {}).get('calibration_content_sha256') != lock.get('calibration_content_sha256')
        or policy.get('settings') != lock['settings']):
        raise ValueError('Selected classical backend timing and calibration lock differ')
    measured = [r for r in benchmark.get('rows', []) if r.get('status') == 'measured'
                and r.get('numerical_equivalence', {}).get('passed')]
    rows = []
    for value in measured:
        row = {key: value[key] for key in ('method', 'device', 'cpu_threads', 'latency_ms_p50',
                                           'latency_ms_p95', 'frames_per_second', 'frames_per_repetition', 'solver_residual_max')}
        row['amortized_median_ms_per_frame'] = row['latency_ms_p50'] / row['frames_per_repetition']
        rows.append(row)
    _old.write_csv(dest / 'corrected_classical_backend_cost.csv', rows)
    _old.write_csv(dest / 'corrected_classical_backend_choices.csv', policy['recommendations'])
    lines = ['\\begin{table*}[t]', '\\centering', '\\scriptsize',
             '\\caption{Classical verification cost on independent calibration observations. '
             'Median wall-clock cost is amortized in ms/frame; throughput is frames/s. '
             'The timed region includes input transfer, fitting, common detection, BEM native detection, '
             'NMSE/BER scoring and scalar transfer. Offline covariance fitting and waveform generation are excluded. '
             'This verification workload is separate from digital receiver deployment latency.}',
             '\\label{tab:corrected_classical_backend}', '\\begin{tabular}{llrrrr}', '\\toprule',
             'Method & Device & CPU threads & Median ms/frame & Frames/s & Maximum residual \\\\', '\\midrule']
    for row in rows:
        lines.append(' & '.join([LABELS[row['method']], row['device'], str(row['cpu_threads']),
                    f'{row["amortized_median_ms_per_frame"]:.3f}', f'{row["frames_per_second"]:.1f}',
                    f'{row["solver_residual_max"]:.1e}']) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    (dest / 'corrected_classical_backend_cost.tex').write_text('\n'.join(lines), encoding='utf-8')
    (dest / 'corrected_classical_cost_insertion.tex').write_text(
        '\\subsection{Classical Verification Throughput}\n'
        'We separately measure the cost of evaluating the fixed classical references, including fitting, '
        'detection and metric calculation on independent calibration observations. CPU thread counts and '
        'CUDA backends use the same complex128 arithmetic and must pass numerical and residual checks. '
        'The fastest qualified backend is used for score verification. This workload differs from the '
        'digital receiver latency measurement.\n'
        '\\input{corrected_results/corrected_classical_backend_cost.tex}\n', encoding='utf-8')
    write_json(dest / 'corrected_classical_cost_scope.json', {
        'protocol': benchmark['protocol'], 'hardware': benchmark.get('hardware'),
        'recommendations': policy['recommendations'], 'benchmark_sha256': file_hash(path)})
    files = ['corrected_classical_backend_cost.csv', 'corrected_classical_backend_choices.csv',
             'corrected_classical_backend_cost.tex', 'corrected_classical_cost_insertion.tex',
             'corrected_classical_cost_scope.json']
    return files, {'status': 'complete', 'benchmark_sha256': file_hash(path)}


def readiness(score_assets_ready, timing_status, timing_assets, classical_cost_status, classical_cost_assets):
    """Score reporting can finish before the two hardware measurements."""
    timing_complete = timing_status.get('status') == 'complete'
    single_ready = timing_complete and 'corrected_digital_receiver_latency.tex' in timing_assets
    throughput_ready = timing_complete and 'corrected_digital_receiver_throughput.tex' in timing_assets
    digital_ready = single_ready
    cost_ready = (classical_cost_status.get('status') == 'complete' and
                  'corrected_classical_backend_cost.tex' in classical_cost_assets)
    pending = []
    if not score_assets_ready:
        pending.append('complete corrected score suite and comparison assets')
    if not single_ready:
        pending.append('single-frame digital receiver CPU/CUDA latency')
    if not throughput_ready:
        pending.append('batched digital receiver CPU/CUDA throughput')
    if not cost_ready:
        pending.append('selected-settings classical verification benchmark')
    return {'score_assets_ready': bool(score_assets_ready),
            'single_frame_latency_ready': bool(single_ready), 'throughput_assets_ready': bool(throughput_ready),
            'digital_latency_ready': bool(digital_ready),
            'classical_verification_cost_ready': bool(cost_ready),
            'paper_ready': bool(score_assets_ready and single_ready and throughput_ready and cost_ready),
            'pending': pending,
            'status': 'complete' if not pending else 'scores_complete_pending_hardware' if score_assets_ready else 'incomplete'}


def write_insertion(dest, macro, tables, suite, latency_available, latency_tables=None):
    lookup = {(r['family'], r['scope'], r['metric'], r['method']): r for r in macro}
    lines = ['% Generated from new corrected baseline scores and reused frozen neural scores.',
             '\\subsection{Independently Calibrated Pilot-Domain References}',
             'We evaluate a pilot-domain separable covariance LMMSE reference and joint occupied-band OCE/DPSS-BEM estimators. '
             'The covariance reference uses normalized sample second moments and the actual transmitted pilot locations. '
             'Its regularization is selected on a held-out part of an independent calibration ensemble. '
             'The BEM order, delay support and ridge are selected using complete independent-calibration solves, '
             'with explicit normal-equation residual checks. The selected settings are fixed for evaluation.',
             'All comparisons use the same received test realizations and reuse the existing three-seed neural scores. '
             'NMSE is averaged in linear units before conversion to dB. Conditions are averaged within each channel, '
             'then channels receive equal weight. Pointwise 95\\% intervals use paired eight-frame generation-cluster '
             'bootstrap and additionally resample the three trained seeds for learned methods.']
    if suite in ('core', 'manuscript', 'all'):
        lines.append('The matched high-mobility comparison includes eleven dynamic profiles at 800, 1200 and 1600 Hz. '
                     'QPSK NMSE uses 15/25 dB, while 16-QAM common-detector BER uses 20 dB; CP length and pilot spacing are 16 and 4.')
        for family, scope, metric in [('nmse_qpsk', 'selected_dynamic', 'nmse'), ('receiver_16qam_20db', 'selected_dynamic', 'ber_diagonal')]:
            methods = ('dopnet',) + tuple(METHODS)
            if all((family, scope, metric, m) in lookup for m in methods):
                unit = 'dB' if metric == 'nmse' else '\\%'
                key, factor = ('mean_db', 1.) if metric == 'nmse' else ('mean', 100.)
                numbers = ', '.join(f'{lookup[(family, scope, metric, m)][key] * factor:.3f}' for m in methods)
                label = 'NMSE' if metric == 'nmse' else 'common-diagonal BER'
                lines.append('For DOP-Net, pilot covariance LMMSE, joint OCE-BEM and joint DPSS-BEM, respectively, '
                             f'the selected-dynamic {label} is {numbers} {unit}.')
    for name in tables:
        if 'native' not in name:
            lines.append('\\input{corrected_results/' + name + '}')
    if 'corrected_core_native_ber.tex' in tables:
        native = ('\\subsection{Estimator-Specific ICI Interfaces}\n'
                  'The recalibrated BEM receivers use full occupied-band ICI matrices. Learned receivers retain '
                  'their five-band interfaces. The statistical covariance reference has no separate ICI interface.\n'
                  '\\input{corrected_results/corrected_core_native_ber.tex}\n')
        (dest / 'corrected_native_insertion.tex').write_text(native, encoding='utf-8')
    latency_table = ('corrected_digital_receiver_latency.tex' in latency_tables if latency_tables is not None
                     else (dest / 'corrected_digital_receiver_latency.tex').is_file())
    throughput_table = ('corrected_digital_receiver_throughput.tex' in latency_tables if latency_tables is not None
                        else (dest / 'corrected_digital_receiver_throughput.tex').is_file())
    if latency_available and (latency_table or throughput_table):
        lines.append('Digital receiver costs are measured separately from the specified input boundary to decoded hard bits, '
                     'including pilot preprocessing, estimation, detection and demapping. Host-origin totals also include '
                     'device transfers. The corresponding tables report measured CPU/CUDA workloads; '
                     'the fastest eligible device is selected separately for each receiver and batch size.')
        if latency_table:
            lines.append('\\input{corrected_results/corrected_digital_receiver_latency.tex}')
        if throughput_table:
            lines.append('\\input{corrected_results/corrected_digital_receiver_throughput.tex}')
    (dest / 'corrected_paper_insertion.tex').write_text('\n\n'.join(lines) + '\n', encoding='utf-8')


def report(corrected_output, reference_results, bootstrap=2000, latency_output=None):
    if bootstrap < 1000:
        raise ValueError('At least 1000 bootstrap draws required for final tables')
    output = Path(corrected_output).resolve()
    evaluation = signed(output / 'evaluation.json')
    lock = signed(output / 'calibration_lock.json')
    lock_hash = file_hash(output / 'calibration_lock.json')
    if (evaluation.get('schema_version') != 2 or evaluation.get('methods') != list(METHODS)
        or evaluation.get('calibration_lock_sha256') != lock_hash or not evaluation.get('streamed_frozen_replay')):
        raise ValueError('Corrected evaluation method/schema/lock/replay contract differs')
    if lock.get('status') != 'locked' or lock.get('methods') != list(METHODS):
        raise ValueError('All corrected methods need an independent locked calibration')
    if lock.get('code_sha256') != evaluation.get('code_sha256'):
        raise ValueError('Calibration/evaluation implementation fingerprints differ')
    validate_calibration_lock(lock, output)
    complete = read_json(output / 'COMPLETE.json')
    for key, expected in [('status', 'complete'), ('signature', evaluation['signature']),
                          ('methods', list(METHODS)), ('calibration_lock_sha256', lock_hash),
                          ('suite', evaluation['suite']), ('source_fingerprints_verified', True)]:
        if complete.get(key) != expected:
            raise ValueError('The declared corrected suite is incomplete or unbound: ' + key)
    reference = Reference(reference_results)
    rows, paired, diagnostics, buckets = [], [], [], defaultdict(lambda: defaultdict(dict))
    rng = np.random.default_rng(920261001)
    seed_draws = rng.integers(0, 3, size=(bootstrap, 3))
    hashes = {}
    try:
        if (evaluation.get('reference_evaluation_signature') != reference.signature
            or evaluation.get('reference_experiment_signature') != reference.experiment['signature']):
            raise ValueError('Corrected/reference experiment provenance differs')
        expected_cells = select_suite(list(reference.metadata.values()), evaluation['suite'])
        expected_ids = {c['cell_id'] for c in expected_cells}
        actual_meta = evaluation.get('cells', [])
        if len(actual_meta) != len(expected_ids) or {c['cell_id'] for c in actual_meta} != expected_ids:
            raise ValueError('Evaluation is not the full declared frozen suite')
        if set(complete['cell_files']) != expected_ids:
            raise ValueError('Root completion must bind every declared cell exactly once')
        validate_required_priors(lock, actual_meta)
        expected_groups = defaultdict(set)
        for cid, meta in reference.metadata.items():
            for group in memberships(meta):
                expected_groups[group].add(cid)
        checked_eval = dict(evaluation, _reference_routes=reference.routes)
        for number, meta in enumerate(actual_meta, 1):
            cid = meta['cell_id']
            if meta != reference.metadata[cid]:
                raise ValueError('Corrected/reference cell metadata differs')
            original = reference.cell(cid)
            scores, native, cluster, diag, score_hash = verify_corrected_cell(
                output, cid, meta, original, reference.cell_hash(cid), checked_eval, lock_hash, complete['cell_files'])
            diagnostics.extend(diag)
            hashes[cid] = score_hash
            cell_rng = np.random.default_rng(int(digest(meta)[:16], 16) ^ 920261001)
            ids, weights = _old.resampling_weights(cluster, bootstrap, cell_rng)
            for metric in ('nmse',) + (('ber_diagonal', 'ber_native') if meta['ber'] else ()):
                values = {}
                for method in LEARNED + OLD_CLASSICAL:
                    idx = reference.indices[method]
                    if metric == 'ber_native' and not np.all(original['native_available'][idx]):
                        continue
                    score = original['nmse'][idx] if metric == 'nmse' else original['errors_' + metric[4:]][idx]
                    denominator = None if metric == 'nmse' else original['bits'][idx]
                    values[method] = _old.metric_samples(score, denominator, cluster, weights, ids)
                for mi, method in enumerate(METHODS):
                    if metric == 'ber_native' and not native[mi]:
                        continue
                    score = scores['nmse'][mi:mi + 1] if metric == 'nmse' else scores['errors_' + metric[4:]][mi:mi + 1]
                    denominator = None if metric == 'nmse' else scores['bits'][mi:mi + 1]
                    values[method] = _old.metric_samples(score, denominator, cluster, weights, ids)
                context = {k: v for k, v in meta.items() if k not in ('studies', 'ber')}
                context.update(suite=evaluation['suite'], frames=512, test_clusters=64)
                for method, (point, boot) in values.items():
                    rows.append(interval_row(context, method, metric, point, boot, seed_draws))
                    if method != 'dopnet':
                        paired.append(paired_row(context, method, metric, point, boot, *values['dopnet'], seed_draws))
                    for family, scope, group_metric in memberships(meta):
                        if group_metric == metric:
                            buckets[(family, scope, metric, method)][meta['channel']][cid] = (point, boot)
            if number % 20 == 0:
                print(f'[corrected-report] verified scores {number}/{len(actual_meta)}', flush=True)
        macro_scores = aggregate_complete_groups(buckets, expected_groups)
        macro, macro_pairs = [], []
        for key, (point, boot, context) in macro_scores.items():
            family, scope, metric, method = key
            context = dict(context, suite=evaluation['suite'])
            # Avoid duplicate metric field when constructing the output row.
            context.pop('metric', None)
            macro.append(interval_row(context, method, metric, point, boot, seed_draws))
            ref = macro_scores.get((family, scope, metric, 'dopnet'))
            if method != 'dopnet' and ref is not None:
                macro_pairs.append(paired_row(context, method, metric, point, boot, ref[0], ref[1], seed_draws))
        dest = output / 'corrected_summary'
        dest.mkdir(parents=True, exist_ok=True)
        csvs = {'cell_summary.csv': rows, 'cell_paired_differences.csv': paired,
                'macro_summary.csv': macro, 'macro_paired_differences.csv': macro_pairs,
                'solver_diagnostics.csv': diagnostics}
        for name, data in csvs.items():
            _old.write_csv(dest / name, data)
        tables = write_tables(dest, macro)
        figures = make_doppler_curves(dest, macro)
        timing_assets, timing_status = latency_assets(dest, latency_output or output / 'latency', reference.experiment['signature'])
        classical_cost_assets, classical_cost_status = classical_backend_assets(dest, output, lock)
        write_insertion(dest, macro, tables, evaluation['suite'], timing_status['status'] == 'complete', timing_assets)
        assets = list(csvs) + tables + figures + timing_assets + classical_cost_assets + ['corrected_paper_insertion.tex']
        if (dest / 'corrected_native_insertion.tex').is_file():
            assets.append('corrected_native_insertion.tex')
        readiness_state = readiness(len(hashes) == len(expected_ids) and bool(tables), timing_status,
                                    timing_assets, classical_cost_status, classical_cost_assets)
        statistics = {'schema_version': 2, **readiness_state, 'suite': evaluation['suite'],
            'expected_cells': len(expected_ids), 'completed_verified_cells': len(hashes),
            'evaluation_signature': evaluation['signature'], 'calibration_lock_sha256': lock_hash,
            'reference_evaluation_signature': reference.signature,
            'bootstrap_repetitions': bootstrap, 'bootstrap_seed': 920261001,
            'intervals': {'learned': CI_LEARNED, 'classical': CI_CLASSICAL},
            'aggregation': 'linear per-frame NMSE; per-cell ratio of bit totals; equal conditions per channel, equal channels',
            'pairing': 'same generation-cluster weights per cell and same global seed draws for all learned estimators',
            'unavailable_statistical_native_ber': 'not evaluated; excluded, never zero',
            'cell_files': hashes, 'calibration_summary': {k: v for k, v in lock.items() if k not in ('signature', 'code_sha256', 'candidates')},
            'latency': timing_status, 'classical_verification_cost': classical_cost_status,
            'reporter_sha256': file_hash(__file__),
            'report_asset_sha256': {name: file_hash(dest / name) for name in assets}}
        write_json(dest / 'statistics.json', statistics)
        (dest / 'README.md').write_text(
            'Corrected references and frozen learned scores\n\n'
            f'Declared suite: {evaluation["suite"]}; {len(hashes)} complete conditions. '
            f'All score and hardware assets ready: {readiness_state["paper_ready"]}. '
            + (('Pending: ' + '; '.join(readiness_state['pending']) + '. ') if readiness_state['pending'] else '') +
            'The new covariance and Cholesky BEM references have separate method IDs and scores. '
            'The historical separable Wiener scores remain a historical reference, not corrected LMMSE results.\n\n'
            'Tables use paired generation-cluster bootstrap, with shared three-seed draws for learned methods. '
            'These intervals differ from the earlier seed-t tables. Native BER is omitted for the covariance reference. '
            'The supplied insertion lists measured values and does not assume which method performs best. '
            'Digital latency tables retain each measured input boundary and include only numerically qualified devices.\n', encoding='utf-8')
        print(f'Saved corrected scores and assets: {dest}; status={readiness_state["status"]}', flush=True)
        return statistics
    finally:
        reference.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corrected-output', required=True)
    parser.add_argument('--reference-results', required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--latency-output')
    args = parser.parse_args()
    report(args.corrected_output, args.reference_results, args.bootstrap, args.latency_output)


if __name__ == '__main__':
    main()
