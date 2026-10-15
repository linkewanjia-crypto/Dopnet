"""Fail-closed reuse of the 27 previously completed, matched checkpoints.

The old archive supplies model weights and per-epoch training-data digests.
New shared caches must reproduce every old tensor except the newly derived
pilot-only view. No old score is copied into a new evaluation.
"""
import hashlib
import io
import json
import copy
import zipfile
from pathlib import Path

from experiment_protocol import SEEDS, VARIANTS, digest, file_hash, make_config, write_json

OLD_VARIANTS = (
    'dopnet', 'cfr_derived', 'slow_time', 'aft_only', 'no_fusion',
    'no_offgrid', 'lee3ddual', 'wimae_ce', 'dopnet_lite',
)
NEW_VARIANTS = tuple(name for name in VARIANTS if name not in OLD_VARIANTS)
NEW_CONFIG_FIELDS = (
    ('data_pipeline', 'emit_pilot_only_fast_time_add'),
    ('train', 'shared_epoch_samples'),
    ('train', 'unique_epoch_samples'),
)


def _archive_member(bundle, suffix):
    matches = [name for name in bundle.namelist() if name.endswith('/' + suffix)]
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one archived {suffix}')
    return matches[0]


def mixed_configs(base, reference):
    """Keep the exact archived configs for old models; train only new configs."""
    old = reference.experiment['configs']
    expected = {f'{variant}_seed{seed}' for variant in OLD_VARIANTS for seed in SEEDS}
    if set(old) != expected:
        raise RuntimeError('Reference checkpoint configuration list differs')
    configs = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            name = f'{variant}_seed{seed}'
            current = make_config(base, variant, seed)
            if variant in OLD_VARIANTS:
                comparable = json.loads(json.dumps(current))
                for section, field in NEW_CONFIG_FIELDS:
                    comparable[section].pop(field, None)
                if comparable != old[name]:
                    raise RuntimeError(f'Old/new training configuration differs: {name}')
                configs[name] = old[name]
            else:
                configs[name] = current
    return configs


def project_archived_checkpoint_config(raw_cfg, variant):
    """Map only documented public-release renames; reject any other drift."""
    cfg = copy.deepcopy(raw_cfg)
    if cfg['system'].get('frame_mode') != 'dual_domain_v4':
        raise RuntimeError('Unknown archived frame-mode label')
    cfg['system']['frame_mode'] = 'dopnet'
    if variant in ('dopnet', 'cfr_derived', 'slow_time', 'aft_only',
                   'no_fusion', 'no_offgrid', 'dopnet_lite'):
        if cfg['model'].get('estimator') != 'v4_dual3d':
            raise RuntimeError('Unknown archived DOP-Net estimator label')
        cfg['model']['estimator'] = 'dopnet'
    if variant == 'dopnet_lite':
        if (cfg['model'].get('ablation_profile') != 'student_v1'
                or cfg.get('distill', {}).get('enabled') is not False):
            raise RuntimeError('Archived Lite checkpoint is not the no-distillation version')
        cfg['model']['ablation_profile'] = 'dopnet_lite'
        del cfg['distill']
    return cfg


class HistoricalTrainingReference:
    def __init__(self, archive, reference):
        self.archive = Path(archive).resolve()
        self.old_code_sha256 = reference.experiment['code_sha256']
        self.old_data_pipeline = reference.experiment['configs']['dopnet_seed42']['data_pipeline']
        self.manifests = {}
        if self.archive.is_dir():
            candidates = []
            for first in self.archive.rglob('core_seed42_epoch001.json'):
                if first.parent.name != 'data_manifests':
                    continue
                try:
                    identity = json.loads(first.read_text(encoding='utf-8'))['identity']
                except (OSError, KeyError, ValueError):
                    continue
                if identity.get('code_sha256') == self.old_code_sha256:
                    candidates.append(first.parent)
            if len(candidates) != 1:
                raise RuntimeError('Expected one matching data_manifests directory '
                                   'under the historical project')
            root = candidates[0]
            self._load_manifests(lambda suffix: (root/Path(suffix).name).read_bytes())
        else:
            with zipfile.ZipFile(str(self.archive)) as bundle:
                self._load_manifests(
                    lambda suffix: bundle.read(_archive_member(bundle, suffix)))
        self.manifest_set_signature = digest({
            f'{seed}_{epoch}': self.manifests[seed, epoch][1]
            for seed in SEEDS for epoch in range(1, 41)
        })

    def _load_manifests(self, read):
        for seed in SEEDS:
            for epoch in range(1, 41):
                records = []
                for group in ('core', 'student'):
                    suffix = (f'data_manifests/{group}_seed{seed}_'
                              f'epoch{epoch:03d}.json')
                    raw = read(suffix)
                    record = json.loads(raw)
                    identity = record['identity']
                    if (identity['code_sha256'] != self.old_code_sha256
                            or identity['seed'] != seed
                            or identity['epoch'] != epoch
                            or identity['samples'] != 12000
                            or identity['batch_size'] != 32
                            or identity['data_pipeline'] != self.old_data_pipeline
                            or record['frames'] != 12000
                            or len(record['batches']) != 375):
                        raise RuntimeError(f'Historical training manifest differs: {suffix}')
                    records.append((record, hashlib.sha256(raw).hexdigest()))
                core, student = records
                if (core[0]['data_signature'] != student[0]['data_signature']
                        or [r['tensor_sha256'] for r in core[0]['batches']]
                        != [r['tensor_sha256'] for r in student[0]['batches']]):
                    raise RuntimeError(f'Old core/Lite streams differ: seed {seed}, epoch {epoch}')
                self.manifests[seed, epoch] = core

    def check_epoch(self, current, seed, epoch):
        old, old_file_sha = self.manifests[seed, epoch]
        identity = json.loads(json.dumps(current['identity']))
        identity['code_sha256'] = self.old_code_sha256
        identity['data_pipeline'].pop('emit_pilot_only_fast_time_add', None)
        # The public release renamed this metadata label; the per-batch tensor
        # hashes below, rather than the label, establish physical equality.
        if identity['system']['frame_mode'] != 'dopnet':
            raise RuntimeError('Unexpected current frame-mode label')
        identity['system']['frame_mode'] = 'dual_domain_v4'
        if identity != old['identity']:
            raise RuntimeError(f'New/old shared epoch identities differ: seed {seed}, epoch {epoch}')
        actual = [r.get('legacy_tensor_sha256') for r in current['batches']]
        expected = [r['tensor_sha256'] for r in old['batches']]
        if len(actual) != 375 or actual != expected:
            first = next((i for i, (a, b) in enumerate(zip(actual, expected))
                          if a != b), None)
            raise RuntimeError(f'New/old training tensors differ: seed {seed}, epoch '
                               f'{epoch}, batch {first}')
        proof = dict(seed=seed, epoch=epoch,
                     historical_manifest_sha256=old_file_sha,
                     historical_data_signature=old['data_signature'],
                     current_data_signature=current['data_signature'],
                     matched_batches=len(actual), matched_frames=12000,
                     comparison='all original tensor hashes, excluding only new pilot-only view')
        proof['signature'] = digest(proof)
        return proof

    def verify_proofs(self, out):
        """All 120 exact-data comparisons must precede combined evaluation."""
        for seed in SEEDS:
            for epoch in range(1, 41):
                path = out/'historical_data_match'/f'seed{seed}_epoch{epoch:03d}.json'
                if not path.is_file():
                    raise RuntimeError(f'Missing old/new training-data proof: {path}')
                proof = json.loads(path.read_text(encoding='utf-8'))
                signature = proof.pop('signature', None)
                expected = self.manifests[seed, epoch]
                if (signature != digest(proof)
                        or proof.get('historical_manifest_sha256') != expected[1]
                        or proof.get('matched_batches') != 375
                        or proof.get('matched_frames') != 12000):
                    raise RuntimeError(f'Old/new training-data proof differs: {path}')


def import_checkpoints(out, reference_path, reference, manifest):
    """Verify archive hashes, checkpoint config/selection, and strict load."""
    from models.receiver import SharedUserChannelEstimator
    from utils.checkpoints import load_checkpoint, save_checkpoint
    import torch

    expected_hashes = reference.evaluation['checkpoint_sha256']
    expected_names = {f'{variant}_seed{seed}' for variant in OLD_VARIANTS for seed in SEEDS}
    if set(expected_hashes) != expected_names:
        raise RuntimeError('Reference evaluation checkpoint list differs')
    with zipfile.ZipFile(str(reference_path)) as bundle:
        for name in sorted(expected_names):
            member = _archive_member(bundle, f'checkpoints/{name}/model.pt')
            raw = bundle.read(member)
            expected = expected_hashes[name]
            if hashlib.sha256(raw).hexdigest() != expected:
                raise RuntimeError(f'Old checkpoint SHA-256 differs: {name}')
            source = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
            variant = name.rsplit('_seed', 1)[0]
            projected = project_archived_checkpoint_config(source['cfg'], variant)
            if (projected != manifest['configs'][name]
                    or source.get('selection_metric') != 'val_nmse_db'):
                raise RuntimeError(f'Old checkpoint configuration/selection differs: {name}')
            model = SharedUserChannelEstimator(projected)
            model.load_state_dict(source['model'], strict=True)
            del model
            folder = out / 'checkpoints' / name
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / 'model.pt'
            if not target.exists():
                normalized = dict(model=source['model'], cfg=projected,
                                  selection_metric='val_nmse_db',
                                  epoch=source.get('epoch'),
                                  source_checkpoint_sha256=expected)
                save_checkpoint(normalized, target)
            checkpoint = load_checkpoint(target, 'cpu')
            if (checkpoint.get('cfg') != projected
                    or checkpoint.get('selection_metric') != 'val_nmse_db'
                    or checkpoint.get('source_checkpoint_sha256') != expected
                    or checkpoint.get('model', {}).keys() != source['model'].keys()
                    or any(not torch.equal(checkpoint['model'][key], value)
                           for key, value in source['model'].items())):
                raise RuntimeError(f'Normalized old checkpoint differs: {name}')
            normalized_hash = file_hash(target)
            marker = dict(signature=manifest['signature'], config=manifest['configs'][name],
                          source_reference_experiment=reference.experiment['signature'],
                          source_checkpoint_sha256=expected,
                          normalized_checkpoint_sha256=normalized_hash)
            marker_path = folder / 'complete.json'
            if marker_path.exists() and json.loads(marker_path.read_text()) != marker:
                raise RuntimeError(f'Old checkpoint completion marker differs: {name}')
            write_json(marker_path, marker)
            del source, checkpoint, raw
