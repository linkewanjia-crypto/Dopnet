"""Publication experiment definitions. No model execution on import."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SEEDS = (42, 43, 44)
TEST_SEED = 730000001
CALIBRATION_SEED = 810000001
VARIANTS = {
    'dopnet': ('dopnet', 'full'),
    'cfr_derived': ('dopnet', 'cfr_derived_add'),
    'slow_time': ('dopnet', 'no_fast_time_add'),
    'pilot_only_fast_time': ('dopnet', 'pilot_only_fast_time'),
    'aft_only': ('dopnet', 'aft_only'),
    'no_fusion': ('dopnet', 'no_physics_aware_fusion'),
    'no_offgrid': ('dopnet', 'no_offgrid_refinement'),
    'lee3ddual': ('lee3ddual', 'full'),
    'wimae_ce': ('wimae_ce', 'full'),
    'dopnet_lite': ('dopnet', 'dopnet_lite'),
    'fortitran': ('fortitran', 'full'),
    'dopnet_25pct': ('dopnet', 'full'),
    'dopnet_50pct': ('dopnet', 'full'),
    'slow_time_25pct': ('dopnet', 'no_fast_time_add'),
    'slow_time_50pct': ('dopnet', 'no_fast_time_add'),
}
STATIC = ('awgn', 'rayleigh', 'rician', 'nakagami')
TRAIN_GPP = ('tdl_b','tdl_c','tdl_d','cdl_b','cdl_c','cdl_d')
HELDOUT = ('tdl_a','tdl_e','cdl_a','cdl_e')
DYNAMIC = ('doubly_selective',) + TRAIN_GPP + HELDOUT
CHANNELS = STATIC + DYNAMIC
CLASSICAL = ('ls_interp', 'lmmse_wiener', 'dpsbem_ls', 'pwb_linear')

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def code_hash(root):
    """Fingerprint runtime physics/training/evaluation code, not derived plots.

    Statistics and figure scripts can be revised without invalidating a
    frozen training manifest. Their own outputs carry a separate analysis
    version and always consume the same immutable raw observations.
    """
    root = Path(root)
    runtime = {
        'experiment_protocol.py', 'run_experiments.py', 'round_robin.py',
        'train.py', 'epoch_cache.py', 'evaluate.py', 'fixed_validation.py',
        'reference_replay.py', 'historical_reuse.py',
        'verify_historical_validation_reconstruction.py',
        'audit_observation_operator.py',
    }
    files = [p for p in root.rglob('*.py')
             if (p.name in runtime and p.parent == root)
             or (p.parent.parent == root and p.parent.name in ('data', 'models', 'utils'))]
    return digest({str(p.relative_to(root)).replace('\\', '/'): file_hash(p)
                   for p in sorted(files)})

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def make_config(base, variant, seed):
    import copy
    cfg = copy.deepcopy(base)
    estimator, profile = VARIANTS[variant]
    cfg['seed'] = seed
    cfg['device'] = 'cuda'
    cfg['model'].update(estimator=estimator, ablation_profile=profile, grad_checkpoint=False)
    full_train_samples = int(base['train']['train_samples'])
    cfg['train']['shared_epoch_samples'] = full_train_samples
    cfg['train']['unique_epoch_samples'] = full_train_samples
    if variant.endswith('_25pct'):
        cfg['train']['unique_epoch_samples'] = full_train_samples // 4
    elif variant.endswith('_50pct'):
        cfg['train']['unique_epoch_samples'] = full_train_samples // 2
    if variant == 'fortitran':
        # Task-matched FortiTran adapter: retain all seven observed pilot times
        # and expose no hidden channel statistics or waveform-only bins.
        cfg['model'].update(fortitran_pilot_pool_f=64, fortitran_pilot_pool_t=7)
    if variant == 'dopnet_lite':
        cfg['model'].update(aft_channels=24, aft_depth=2, add_channels=16, add_depth=2,
            dropout=.05, aft_token_dim=8, aft_context_channels=1, aft_axial_blocks=0,
            aft_gat_hidden=8, aft_gat_heads=1, add_token_dim=8, add_context_channels=1,
            add_axial_blocks=0, add_gat_hidden=8, add_gat_heads=1, gate_hidden=8, fusion_hidden=8)
        cfg['train']['doppler_offset_l2_weight'] = 0.
    cfg['train'].update(validation_seed=610000001, baseline_loss_mode='unified',
                        cudnn_benchmark=False, tf32=False, fused_optimizer=False,
                        validation_cache=True, batch_size=32)
    # All variants use identical simulation chunk sizes, pilots and draws.
    cfg['data_pipeline'].update(generation_batch_size=32, microbatch_size=4,
        synthetic_microbatch_size=4, threegpp_microbatch_size=4,
        emit_pilot_ls=True, emit_slow_time_add=True,
        emit_pilot_only_fast_time_add=True, shuffle_buffer=False,
        require_cuda=True)
    # Keep Sionna link chunking fixed across all variants and seeds.  The
    # requested protocol value is 8; changing it later requires a new manifest.
    cfg['system']['threegpp_link_batch_size'] = 8
    from data.channels import validate_channel_config
    validate_channel_config(cfg['system'])
    return cfg

def experiment_cells(cfg):
    """Fixed grids; no selecting test conditions after looking at results."""
    from data.frame_feasibility import feasible_cp_choices, feasible_pilot_spacings
    cells = {}
    skipped = []
    def add(ch, fd, snr, mod='qpsk', cp=16, spacing=4, ds=300., study='nmse', ber=False):
        meta = dict(channel=ch, doppler_hz=float(fd), snr_db=float(snr), modulation=mod,
                    cp_len=cp, pilot_spacing=spacing, delay_spread_ns=float(ds))
        if cp not in feasible_cp_choices(cfg, ds, ch) or spacing not in feasible_pilot_spacings(cfg,cp):
            skipped.append(dict(**meta, study=study, reason='infeasible; no geometry substitution'))
            return
        key = digest(meta)[:16]
        if key not in cells:
            cells[key] = dict(**meta, cell_id=key, studies=[], ber=False)
        if study not in cells[key]['studies']:
            cells[key]['studies'].append(study)
        cells[key]['ber'] |= ber
    for ch in CHANNELS:
        for fd in ((0,) if ch in STATIC else (0,400,800,1200,1600)):
            for snr in (5,15,25,35): add(ch,fd,snr)
    for ch in DYNAMIC:
        for fd in (0,400,800,1200,1600):
            for mod in ('bpsk','qpsk','16qam','64qam'):
                add(ch,fd,20,mod,study='receiver',ber=True)
            for snr in (10,30): add(ch,fd,snr,'16qam',study='receiver_snr',ber=True)
    for ch in ('doubly_selective','tdl_b','cdl_c','tdl_a','cdl_e'):
        for fd in (600,1000,1400): add(ch,fd,20,study='unseen_doppler',ber=True)
        for ds in (100,700):
            for fd in (800,1600):
                add(ch,fd,20,cp=32,ds=ds,study='delay_shift',ber=True)
        for cp in (8,16,32):
            for spacing in (2,4,8):
                add(ch,1200,20,cp=cp,spacing=spacing,study='pilot_cp',ber=True)
    return list(cells.values()), skipped
