"""Strict replay contract for the public fixed-test realization archive.

Metadata inspection is CPU-only.  ``--preflight`` regenerates three complete
cells on the CUDA server; the evaluator subsequently verifies every cell's
waveform fingerprint before it can save scores.  No archived score is used as a
model prediction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from experiment_protocol import TEST_SEED, digest, experiment_cells

EXPECTED_CELLS = 636
EXPECTED_SAMPLES = 512
EXPECTED_BATCH = 8
# The public archive recorded these six tensors.  New optional input features
# may be generated for new ablations, but are not appended to this old digest.
REFERENCE_WAVEFORM_KEYS = (
    'input', 'add_input', 'add_input_slow', 'rx_data', 'target_h', 'tx_label'
)
_HEX64 = re.compile(r'^[0-9a-f]{64}$')


def _as_scalar(value, name):
    if np.asarray(value).shape != ():
        raise RuntimeError(f'Reference {name} must be a scalar')
    return str(np.asarray(value).item())


class ReferenceResults:
    """Load the full public test contract from a directory or release ZIP."""

    def __init__(self, location):
        self.location = Path(location).expanduser().resolve()
        if not self.location.exists():
            raise FileNotFoundError(f'Reference results not found: {self.location}')
        self._zip = None
        if self.location.is_file():
            self._zip = zipfile.ZipFile(self.location)
            candidates = [name for name in self._zip.namelist()
                          if name.endswith('/evaluation.json') or name == 'evaluation.json']
            if len(candidates) != 1:
                self._zip.close()
                raise RuntimeError('Reference ZIP must contain exactly one evaluation.json')
            self._prefix = candidates[0][:-len('evaluation.json')]
        else:
            root = self.location
            if not (root / 'evaluation.json').is_file():
                candidates = list(root.glob('*/evaluation.json'))
                if len(candidates) != 1:
                    raise RuntimeError('Reference directory must contain exactly one evaluation.json')
                root = candidates[0].parent
            self._root = root
        try:
            self.evaluation = json.loads(self._read('evaluation.json'))
            self.experiment = json.loads(self._read('experiment.json'))
            self._validate_metadata()
            self.cell_fingerprints = self._read_cell_fingerprints()
            self.fingerprint = digest(dict(
                evaluation_signature=self.evaluation['signature'],
                experiment_signature=self.experiment['signature'],
                waveform_keys=REFERENCE_WAVEFORM_KEYS,
                realization_sha256=self.cell_fingerprints))
        finally:
            if self._zip is not None:
                self._zip.close()
                self._zip = None

    def _read(self, name):
        if self._zip is not None:
            return self._zip.read(self._prefix + name)
        return (self._root / name).read_bytes()

    def _validate_metadata(self):
        for name, obj in [('experiment.json', self.experiment),
                          ('evaluation.json', self.evaluation)]:
            recorded = obj.get('signature')
            calculated = digest({key: value for key, value in obj.items()
                                 if key != 'signature'})
            if recorded != calculated:
                raise RuntimeError(f'Reference {name} has an invalid signature')
        if self.evaluation.get('experiment_signature') != self.experiment['signature']:
            raise RuntimeError('Reference evaluation and experiment signatures disagree')
        if self.evaluation.get('test_seed') != TEST_SEED:
            raise RuntimeError(f'Reference test seed is not {TEST_SEED}')
        if self.experiment.get('samples_per_cell') != EXPECTED_SAMPLES:
            raise RuntimeError(f'Reference must contain {EXPECTED_SAMPLES} samples per cell')
        if self.experiment.get('batch_size') != EXPECTED_BATCH:
            raise RuntimeError(f'Reference must use test batch size {EXPECTED_BATCH}')
        cells = self.evaluation.get('cells', [])
        if len(cells) != EXPECTED_CELLS:
            raise RuntimeError(f'Reference must contain {EXPECTED_CELLS} cells')
        ids = [cell['cell_id'] for cell in cells]
        if len(set(ids)) != EXPECTED_CELLS:
            raise RuntimeError('Reference contains duplicate cell IDs')
        if not self.evaluation.get('routes'):
            raise RuntimeError('Reference contains no evaluation routes')

    def _read_cell_fingerprints(self):
        result = {}
        route_count = len(self.evaluation['routes'])
        expected_clusters = np.arange(EXPECTED_SAMPLES) // EXPECTED_BATCH
        for cell in self.evaluation['cells']:
            cell_id = cell['cell_id']
            try:
                with np.load(io.BytesIO(self._read(f'cells/{cell_id}.npz')),
                             allow_pickle=False) as archive:
                    if _as_scalar(archive['signature'], 'cell signature') != self.evaluation['signature']:
                        raise RuntimeError(f'Reference cell signature differs: {cell_id}')
                    if json.loads(_as_scalar(archive['meta'], 'cell metadata')) != cell:
                        raise RuntimeError(f'Reference cell metadata differs: {cell_id}')
                    if archive['nmse'].shape != (route_count, EXPECTED_SAMPLES):
                        raise RuntimeError(f'Reference score shape differs: {cell_id}')
                    if not np.array_equal(archive['cluster_id'], expected_clusters):
                        raise RuntimeError(f'Reference 512/8 cluster layout differs: {cell_id}')
                    fingerprint = _as_scalar(archive['realization_sha256'], 'waveform fingerprint')
            except (KeyError, FileNotFoundError) as error:
                raise RuntimeError(f'Reference cell missing or incomplete: {cell_id}') from error
            if not _HEX64.fullmatch(fingerprint):
                raise RuntimeError(f'Invalid reference waveform SHA-256: {cell_id}')
            result[cell_id] = fingerprint
        return result

    def validate_protocol(self, manifest, cfg, cells, skipped):
        """Reject an altered test grid, generator configuration, or 512/8 draw."""
        if manifest.get('samples_per_cell') != EXPECTED_SAMPLES or manifest.get('batch_size') != EXPECTED_BATCH:
            raise RuntimeError('Current test protocol must use 512 samples per cell and batch size 8')
        if cells != self.evaluation['cells']:
            raise RuntimeError('Current 636 test-cell specifications/order differ from the reference')
        if skipped != self.evaluation['skipped_infeasible']:
            raise RuntimeError('Current infeasible-cell record differs from the reference')
        source_cfg = self.experiment['configs']['dopnet_seed42']
        if cfg['system'] != source_cfg['system']:
            raise RuntimeError('Current system configuration differs from fixed-test reference')
        current_pipeline = copy.deepcopy(cfg['data_pipeline'])
        current_pipeline.pop('emit_pilot_only_fast_time_add', None)
        if current_pipeline != source_cfg['data_pipeline']:
            raise RuntimeError('Current data pipeline differs from fixed-test reference')

    def assert_fingerprint(self, cell, observed):
        cell_id = cell['cell_id'] if isinstance(cell, dict) else cell
        expected = self.cell_fingerprints[cell_id]
        if observed != expected:
            raise RuntimeError(
                f'Fixed-test waveform mismatch in cell {cell_id}: '
                f'generated {observed}, archive {expected}. '
                'Do not use this cell; inspect simulator version and RNG before training/evaluation.')

    def provenance(self):
        return dict(reference_evaluation_signature=self.evaluation['signature'],
                    reference_experiment_signature=self.experiment['signature'],
                    reference_realizations_sha256=self.fingerprint,
                    reference_cells=EXPECTED_CELLS,
                    reference_samples_per_cell=EXPECTED_SAMPLES,
                    reference_batch_size=EXPECTED_BATCH,
                    reference_waveform_keys=list(REFERENCE_WAVEFORM_KEYS))


def update_waveform_digest(hasher, batch):
    """Hash exact archived tensor byte order; no model outputs enter this hash."""
    for name in REFERENCE_WAVEFORM_KEYS:
        hasher.update(batch[name].contiguous().cpu().numpy().tobytes())


def preflight(reference_path, config_path='configs/dopnet.yaml'):
    """Server CUDA check of representative complete archived cells, before training."""
    import torch
    from train import load_cfg
    from data.generator import BatchChannelGenerator
    from experiment_protocol import make_config

    if not torch.cuda.is_available():
        raise RuntimeError('Reference waveform preflight requires the CUDA training server')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    reference = ReferenceResults(reference_path)
    cfg = make_config(load_cfg(config_path), 'dopnet', 42)
    cfg['model']['ablation_profile'] = 'full'
    cfg['data_pipeline'].update(emit_pilot_ls=True, emit_slow_time_add=True)
    cells, skipped = experiment_cells(cfg)
    manifest = dict(samples_per_cell=EXPECTED_SAMPLES, batch_size=EXPECTED_BATCH)
    reference.validate_protocol(manifest, cfg, cells, skipped)
    chosen = []
    for channel, study in [('awgn', 'nmse'),
                           ('doubly_selective', 'receiver'),
                           ('cdl_e', 'pilot_cp')]:
        selected = next((cell for cell in cells
                         if cell['channel'] == channel and study in cell['studies']), None)
        if selected is None:
            raise RuntimeError(f'No preflight cell for {channel}/{study}')
        chosen.append(selected)
    device = torch.device('cuda:0')
    gen = BatchChannelGenerator(cfg, device, TEST_SEED)
    with torch.inference_mode():
        for cell in chosen:
            hasher = hashlib.sha256()
            condition = dict(channel_type=cell['channel'], modulation=cell['modulation'],
                             delay_spread_ns=cell['delay_spread_ns'],
                             doppler_hz=cell['doppler_hz'], cp_len=cell['cp_len'],
                             pilot_spacing=cell['pilot_spacing'])
            for start in range(0, EXPECTED_SAMPLES, EXPECTED_BATCH):
                seed = TEST_SEED + int(digest(dict(cell=cell['cell_id'], offset=start))[:15], 16)
                rng = torch.Generator(device=device).manual_seed(seed)
                batch = gen.simulate(EXPECTED_BATCH, condition,
                    torch.full((EXPECTED_BATCH,), cell['snr_db'], device=device),
                    rng, seed + 1000000001)
                update_waveform_digest(hasher, batch)
            reference.assert_fingerprint(cell, hasher.hexdigest())
            print(f"Reference replay PASS: {cell['cell_id']} {cell['channel']}", flush=True)
    print('REFERENCE PREFLIGHT PASSED: three complete 512-frame cells', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-results', required=True,
                        help='DOPNet_Results.zip or its extracted directory')
    parser.add_argument('--config', default='configs/dopnet.yaml')
    parser.add_argument('--preflight', action='store_true',
                        help='Regenerate representative cells on the CUDA server')
    args = parser.parse_args()
    if args.preflight:
        preflight(args.reference_results, args.config)
    else:
        reference = ReferenceResults(args.reference_results)
        print(json.dumps(reference.provenance(), indent=2))
