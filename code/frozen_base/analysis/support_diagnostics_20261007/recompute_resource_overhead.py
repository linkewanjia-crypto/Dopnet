"""Recompute hardware-independent frame accounting from the saved configuration."""
from __future__ import annotations
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parents[4] / 'code/training/main/configs/dopnet.yaml')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    # A small, explicit dependency is used only for this configuration reader.
    import yaml
    config = args.config.resolve()
    out = args.output_dir.resolve()
    if out == config.parent or out.is_relative_to(config.parent):
        parser.error('Output must be separate from the source configuration directory.')
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory must be new or empty.')
    s = yaml.safe_load(config.read_text(encoding='utf-8'))['system']
    N = int(s['n_subcarriers'])
    nfft = int(s['waveform_fft_size'])
    T = int(s['frame_num_symbols'])
    period = int(s['pilot_symbol_period'])
    U = int(s['n_users'])
    df = float(s['subcarrier_spacing_hz'])
    pilots = list(range(0, T, period))
    P, D = len(pilots), T - len(pilots)
    rows = []
    for cp in s['cp_len_choices']:
        for spacing in s['pilot_spacing_choices']:
            if spacing < U or spacing * cp > N:
                continue
            cp_useful = Fraction(N, N + cp)
            pilot_user = Fraction(P, T * spacing)
            union = U * pilot_user
            symbol_ms = 1000 * (1 + cp / N) / df
            rows.append({
                'cp_active': cp, 'cp_physical': cp * nfft // N, 'spacing': spacing,
                'visible_symbols': T, 'pilot_only_symbols': P, 'data_symbols': D,
                'user_nonzero_pilot_REs': P * N // spacing,
                'joint_nonzero_pilot_REs': U * P * N // spacing,
                'pilot_only_time_fraction': float(Fraction(P, T)),
                'per_user_nonzero_pilot_RE_fraction': float(pilot_user),
                'joint_nonzero_pilot_RE_fraction': float(union),
                'joint_null_RE_in_pilot_blocks_fraction': float(Fraction(P, T) - union),
                'visible_payload_RE_fraction': float(Fraction(D, T)),
                'cp_useful_time_fraction': float(cp_useful),
                'visible_payload_time_coefficient': float(Fraction(D, T) * cp_useful),
                'visible_frame_duration_ms': T * symbol_ms,
                'simulated_27_block_duration_ms': (T + 2) * symbol_ms,
                'if_2_boundary_blocks_charged_payload_time_coefficient': float(Fraction(D, T + 2) * cp_useful),
                'bracketing_pilot_max_lookahead_ms': (period - 1) * symbol_ms,
                'PWB_last_pilot_lookahead_from_first_data_ms': (pilots[-1] - 1) * symbol_ms,
            })
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'resource_overhead.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        'source_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
        'geometry_pairs': len(rows), 'pilot_only_time_fraction': P / T,
        'rows': rows,
        'scope': 'Frame resource accounting and scheduling lookahead; no measured compute latency or coded goodput.',
        'training_or_inference': False,
    }
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'geometry_pairs': len(rows), 'pilot_only_time_fraction': P / T}))


if __name__ == '__main__':
    main()
