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

from experiment_protocol import SEEDS, VARIANTS, make_config, code_hash, digest, write_json

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='results/dopnet')
    ap.add_argument('--samples-per-cell', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=8, help='Evaluation generation cluster size; fixed by manifest')
    ap.add_argument('--stage', choices=['all','train','evaluate','aggregate','preflight'], default='all')
    ap.add_argument('--bootstrap', type=int, default=2000)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    out = Path(args.output).resolve()
    if args.stage == 'aggregate':
        subprocess.run([sys.executable,'aggregate_results.py','--output',str(out),
                        '--bootstrap',str(args.bootstrap)],check=True)
        return
    # Deliberately refuse a CPU run; this entry point belongs on the training server.
    import torch
    import yaml
    from train import load_cfg
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required. Do not run experiments on the laptop.')
    if args.stage == 'preflight':
        subprocess.run([sys.executable,'server_preflight.py'],check=True)
        subprocess.run([sys.executable,'validate_round_robin.py'],check=True)
        return
    if args.samples_per_cell < 128 or args.samples_per_cell % args.batch_size:
        raise SystemExit('Use >=128 samples and an exact multiple of evaluation batch size.')
    if args.samples_per_cell // args.batch_size < 32:
        raise SystemExit('At least 32 independent generation clusters per cell are required.')
    out.mkdir(parents=True, exist_ok=True)
    base = load_cfg('configs/dopnet.yaml')
    configs = {f'{v}_seed{s}': make_config(base,v,s) for v in VARIANTS for s in SEEDS}
    payload = dict(code_sha256=code_hash(root), configs=configs,
                   samples_per_cell=args.samples_per_cell, batch_size=args.batch_size,
                   torch=torch.__version__, cuda=torch.version.cuda,
                   packages={p:importlib.metadata.version(p) for p in ('sionna','numpy','scipy','PyYAML','matplotlib','pandas')},
                   gpu=torch.cuda.get_device_name(), python=sys.version,
                   cuda_allocator_conf=os.environ.get('PYTORCH_CUDA_ALLOC_CONF',''))
    manifest = dict(**payload, signature=digest(payload))
    path = out/'experiment.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
        raise SystemExit('Manifest changed. Use a new --output; old/new runs must not be mixed.')
    write_json(path,manifest)
    (out/'configs').mkdir(exist_ok=True)
    for name,cfg in configs.items():
        (out/'configs'/f'{name}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
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
    if args.stage in ('all','train'):
        from round_robin import train_rounds
        train_rounds(out,configs,manifest,run)
    if args.stage in ('all','evaluate'):
        run([sys.executable,'evaluate.py','--output',out], 'evaluation')
        run([sys.executable,'aggregate_results.py','--output',out,'--bootstrap',args.bootstrap], 'aggregation')
        run([sys.executable,'plot_results.py','--output',out], 'plots')
        run([sys.executable,'profile_models.py','--output',out], 'profiling')
        run([sys.executable,'profile_receiver_latency.py','--output',out], 'receiver_latency')
    print(f'Finished requested stage; outputs: {out}')

if __name__ == '__main__':
    # Acquire before main writes a manifest or starts any subprocess.
    from experiment_lock import experiment_lock
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',default='results/dopnet')
    early,_=parser.parse_known_args()
    output=Path(early.output)
    if not output.is_absolute(): output=Path(__file__).resolve().parent/output
    with experiment_lock(output): main()
