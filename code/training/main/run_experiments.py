"""Run the complete DOP-Net experiment pipeline on a CUDA server."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.metadata
from utils.cuda_memory import configure_cuda_allocator

# All child training/cache/evaluation processes inherit this allocator
# policy, so the documented server command stays unchanged.
configure_cuda_allocator()

from experiment_protocol import SEEDS, VARIANTS, make_config, code_hash, digest, file_hash, write_json
from reference_replay import ReferenceResults

REFERENCE_RELEASE_SHA256 = '3ff13781fe02e2637f8a5654285b883e2bdf2a1aa1889dc05cdecf116ae9a9a4'

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='results/dopnet')
    ap.add_argument('--samples-per-cell', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=8, help='Evaluation generation cluster size; fixed by manifest')
    ap.add_argument('--stage', choices=['all','train','evaluate','aggregate','preflight'], default='all')
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--reference-results', type=Path,
                    help='The immutable DOPNet_Results.zip used for all 636 test realizations')
    ap.add_argument('--fixed-validation-cache', type=Path,
                    help='Extracted historical 2,000-frame validation cache')
    ap.add_argument('--fixed-validation-sha256',
                    default='83d0e03524628b617671c42a73d3c458cc3395c31965a4e22295e8389f56e597')
    ap.add_argument('--reuse-existing', action='store_true',
                    help='Reuse the 27 matched archived checkpoints and train the retained additions')
    ap.add_argument('--historical-training-archive', type=Path,
                    help='Original DOPNet.zip or extracted project with 120 core/Lite epoch manifests')
    ap.add_argument('--historical-data-reference', type=Path,
                    help='Complete original 120-epoch data manifests; fresh training never imports old weights')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    out = Path(args.output).resolve()
    if args.stage == 'aggregate':
        subprocess.run([sys.executable,'aggregate_results.py','--output',str(out),
                        '--bootstrap',str(args.bootstrap)],check=True)
        return
    if args.reference_results is None:
        raise SystemExit('--reference-results is required; use the supplied DOPNet_Results.zip')
    reference_path=args.reference_results.resolve()
    if not reference_path.is_file() or reference_path.suffix.lower() != '.zip':
        raise SystemExit('Use the supplied DOPNet_Results.zip file, not a directory')
    if file_hash(reference_path) != REFERENCE_RELEASE_SHA256:
        raise SystemExit('Reference results ZIP SHA-256 differs from the supplied public release')
    reference=ReferenceResults(reference_path)
    if args.samples_per_cell != 512 or args.batch_size != 8:
        raise SystemExit('The fixed reference contains exactly 512 frames per cell in batches of 8')
    if args.fixed_validation_cache is None:
        raise SystemExit('--fixed-validation-cache is required; validation must not be regenerated')
    if args.reuse_existing != (args.historical_training_archive is not None):
        raise SystemExit('Use --reuse-existing together with --historical-training-archive')
    if args.reuse_existing:
        data_reference=args.historical_training_archive.resolve()
        if (args.historical_data_reference is not None
                and args.historical_data_reference.resolve() != data_reference):
            raise SystemExit('Historical data references disagree; do not mix archives')
    else:
        if args.historical_data_reference is None:
            raise SystemExit('--historical-data-reference is required for fresh training; every old data batch must match before updates')
        data_reference=args.historical_data_reference.resolve()
    validation_path=args.fixed_validation_cache.resolve()
    # Deliberately refuse a CPU run; this entry point belongs on the training server.
    import torch
    import yaml
    from train import load_cfg
    from fixed_validation import (HISTORICAL_CACHE_SHA256,
        assert_historical_identity, assert_reconstruction_attestation,
        verify_historical_cache_file)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required. Do not run experiments on the laptop.')
    if args.fixed_validation_sha256.lower() != HISTORICAL_CACHE_SHA256:
        raise SystemExit('Unexpected fixed validation SHA-256')
    base = load_cfg('configs/dopnet.yaml')
    assert_historical_identity(make_config(base,'dopnet',42))
    verify_historical_cache_file(validation_path,HISTORICAL_CACHE_SHA256)
    try:
        assert_reconstruction_attestation(make_config(base,'dopnet',42),validation_path)
    except (FileNotFoundError,RuntimeError):
        subprocess.run([sys.executable,'verify_historical_validation_reconstruction.py',
                        '--cache',str(validation_path),'--sha256',HISTORICAL_CACHE_SHA256],check=True)
        assert_reconstruction_attestation(make_config(base,'dopnet',42),validation_path)
    if args.stage == 'preflight':
        subprocess.run([sys.executable,'server_preflight.py'],check=True)
        subprocess.run([sys.executable,'validate_round_robin.py'],check=True)
        subprocess.run([sys.executable,'reference_replay.py','--reference-results',
                        str(reference_path),'--preflight'],check=True)
        return
    if args.samples_per_cell < 128 or args.samples_per_cell % args.batch_size:
        raise SystemExit('Use >=128 samples and an exact multiple of evaluation batch size.')
    if args.samples_per_cell // args.batch_size < 32:
        raise SystemExit('At least 32 independent generation clusters per cell are required.')
    out.mkdir(parents=True, exist_ok=True)
    from historical_reuse import (HistoricalTrainingReference, mixed_configs,
                                  import_checkpoints, NEW_VARIANTS, OLD_VARIANTS)
    historical_training=HistoricalTrainingReference(data_reference,reference)
    if args.reuse_existing:
        configs=mixed_configs(base,reference)
        training_mode=f'reuse_{len(OLD_VARIANTS)*len(SEEDS)}_train_{len(NEW_VARIANTS)*len(SEEDS)}'
    else:
        configs = {f'{v}_seed{s}': make_config(base,v,s) for v in VARIANTS for s in SEEDS}
        training_mode=f'retrain_{len(configs)}'
    payload = dict(code_sha256=code_hash(root), configs=configs,
                   training_mode=training_mode,
                   historical_data_comparison_required=True,
                   historical_training_manifest_signature=historical_training.manifest_set_signature,
                   samples_per_cell=args.samples_per_cell, batch_size=args.batch_size,
                   validation_source='historical_fixed_cache',
                   validation_cache_sha256=HISTORICAL_CACHE_SHA256,
                   reference_results=reference.provenance(),
                   reference_archive_sha256=REFERENCE_RELEASE_SHA256,
                   torch=torch.__version__, cuda=torch.version.cuda,
                   packages={p:importlib.metadata.version(p) for p in ('sionna','numpy','scipy','PyYAML','matplotlib','pandas')},
                   gpu=torch.cuda.get_device_name(), python=sys.version,
                   cuda_allocator_conf=os.environ.get('PYTORCH_CUDA_ALLOC_CONF',''))
    if args.reuse_existing:
        payload['reused_checkpoint_sha256']=reference.evaluation['checkpoint_sha256']
    manifest = dict(**payload, signature=digest(payload))
    path = out/'experiment.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
        raise SystemExit('Manifest changed. Use a new --output; old/new runs must not be mixed.')
    write_json(path,manifest)
    (out/'configs').mkdir(exist_ok=True)
    for name,cfg in configs.items():
        (out/'configs'/f'{name}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    if args.reuse_existing:
        for seed in SEEDS:
            (out/'configs'/f'cache_seed{seed}.yaml').write_text(
                yaml.safe_dump(make_config(base,'dopnet',seed),sort_keys=False),encoding='utf-8')
    def run(command, name):
        log=out/'logs'/f'{name}.log'
        log.parent.mkdir(exist_ok=True)
        print('\n>>>', ' '.join(map(str,command)),flush=True)
        # Keep tqdm carriage-return redraws intact. Text-mode line iteration
        # converts '\r' redraws into separate lines; binary passthrough does not.
        with log.open('ab') as f:
            proc=subprocess.Popen(
                list(map(str,command)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            while True:
                chunk=proc.stdout.read(4096)
                if not chunk:
                    break
                f.write(chunk); f.flush()
                stream=getattr(sys.stdout,'buffer',None)
                if stream is not None:
                    stream.write(chunk); stream.flush()
                else:
                    sys.stdout.write(chunk.decode('utf-8',errors='replace'))
                    sys.stdout.flush()
            if proc.wait(): raise RuntimeError(f'Failed: see {log}')
    # Mandatory tests run on the GPU server, never invoked during packaging.
    run([sys.executable,'server_preflight.py'], 'preflight')
    run([sys.executable,'validate_round_robin.py'], 'round_robin_validation')
    run([sys.executable,'reference_replay.py','--reference-results',reference_path,
         '--preflight'], 'fixed_test_replay_preflight')
    run([sys.executable,'audit_observation_operator.py','--output',out/'operator_audit.json'], 'operator_audit')
    if args.reuse_existing:
        import_checkpoints(out,reference_path,reference,manifest)
    if args.stage in ('all','train'):
        from round_robin import train_rounds
        train_rounds(out,configs,manifest,run,
                     fixed_validation_cache=validation_path,
                     fixed_validation_sha256=HISTORICAL_CACHE_SHA256,
                     train_variants=NEW_VARIANTS if args.reuse_existing else None,
                     cache_config_prefix='cache' if args.reuse_existing else 'dopnet',
                     historical_training_reference=historical_training)
    if args.stage in ('all','evaluate'):
        historical_training.verify_proofs(out)
        run([sys.executable,'evaluate.py','--output',out,
             '--reference-results',reference_path], 'evaluation')
        run([sys.executable,'aggregate_results.py','--output',out,'--bootstrap',args.bootstrap], 'aggregation')
        run([sys.executable,'plot_results.py','--output',out], 'plots')
        run([sys.executable,'profile_models.py','--output',out], 'profiling')
        run([sys.executable,'profile_receiver_latency.py','--output',out],
            'receiver_latency')
        run([sys.executable,'submission_report.py','--output',out,
             '--reference-results',reference_path], 'submission_report')
    print(f'Finished requested stage; outputs: {out}')

if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        main()
        raise SystemExit(0)
    # Acquire before main writes a manifest or starts any subprocess.
    from experiment_lock import experiment_lock
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',default='results/dopnet')
    early,_=parser.parse_known_args()
    output=Path(early.output)
    if not output.is_absolute(): output=Path(__file__).resolve().parent/output
    with experiment_lock(output): main()
