"""Serial CUDA hardware qualification only; no waveform/train/validation data.

An administrative parent-delivery receipt is required. This is not evidence
of real waveform throughput, exact full-path peak memory, or scientific scores.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from .artifacts import digest, file_hash, read_signed, sign, write_json
from .prepare import validate_manifest
from .train import (check_deadline, environment, load_runtime, online_step,
                    branch_tensor_hash, adam, utc_now)


def gate_parents(receipt_path):
    receipt = read_signed(receipt_path)
    if receipt.get('status') != 'original_forti_wimae_full_delivery_verified_all_workers_exited':
        raise ValueError('Parents have not actually finished and passed delivery checkers')
    if set(receipt.get('archives', {})) != {'original', 'forti', 'wimae'}:
        raise ValueError('Incomplete parent archive receipt')
    for label, record in receipt['archives'].items():
        path = Path(record['path'])
        if (record.get('full_checker_passed') is not True or file_hash(path) != record['sha256'] or
                path.stat().st_size != record['bytes']):
            raise ValueError('Parent archive receipt invalid: ' + label)
    for pid in receipt.get('exited_pids', []):
        if type(pid) is not int or Path(f'/proc/{pid}').exists():
            raise ValueError('Parent worker/controller still exists')
    # Also inspect all processes; a historical PID alone is insufficient.
    forbidden = ('dl_baseline_expansion.queue_wimae', 'dl_baseline_expansion.evaluate_wimae',
                 'dl_baseline_expansion.train_wimae', 'budget_matched_baselines.queue',
                 'dl_baseline_expansion.report_wimae',
                 'new_baseline_latency_v1.run', 'lee_meta_equal40.train', 'lee_meta_equal40.probe')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name) == os.getpid():
            continue
        try:
            words = path.read_bytes().split(b'\0')
            joined = ' '.join(w.decode(errors='replace') for w in words if w)
        except (OSError, ValueError):
            continue
        if any(word in joined for word in forbidden):
            raise ValueError('Active scientific/probe task; serial qualification is required')
    query = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free',
                            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    lines = query.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError('Expected one original GPU')
    total, free = map(int, lines[0].split(','))
    if total < 47000 or free < int(37.5 * 1024):
        raise ValueError('Insufficient original3GPP +3GiB headroom before probe')
    return receipt, {'total_mib': total, 'free_mib_before': free}


def chain_rule_scalar_gate(torch):
    from torch.utils.checkpoint import checkpoint
    theta = torch.tensor(0.7, device='cuda:0', requires_grad=True)
    support = checkpoint(lambda t: (t-1.2).square()/2, theta, use_reentrant=False)
    inner = torch.autograd.grad(support, theta, create_graph=True)[0]
    phi = theta-0.2*inner
    query = checkpoint(lambda p: (p+0.5).square()/2, phi, use_reentrant=False)
    full = float(torch.autograd.grad(query, theta)[0])
    if abs(full-1.04) > 2e-6 or abs(full-1.3) < 0.1:
        raise ValueError('Actual CUDA checkpointed full chain rule failed')
    return full


def probe(args):
    if os.name != 'posix':
        raise ValueError('Run probes only through authenticated server terminal')
    check_deadline()
    manifest, cfg, _ = validate_manifest(args.manifest, args.project, 42)
    parent, hardware = gate_parents(args.parent_receipt)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Do not overwrite or repeat a recorded probe')
    result = {'schema': 'lee_actual_double_backward_probe_v1',
              'status': 'failed', 'manifest_signature': manifest['signature'],
              'implementation_sha256': manifest['implementation_sha256'],
              'recipe_sha256': cfg['lee_meta_reproduction']['recipe_sha256'],
              'microbatch': args.microbatch, 'parent_receipt_sha256': file_hash(args.parent_receipt),
              'hardware': hardware, 'synthetic_hardware_only': True,
              'scientific_data_generated': False, 'validation_cache_opened': False,
              'fullpath_qualification': False, 'exact_fullpath_peak_verified': False,
              'training_started': False, 'observed_start_utc': utc_now().isoformat()}
    try:
        torch, _, _, module, updates = load_runtime(args.project, hardware_only=True)
        result['environment'] = environment(torch)
        result['scalar_full_chain_rule_gradient'] = chain_rule_scalar_gate(torch)
        generator = torch.Generator(device='cuda:0').manual_seed(970000043)
        inputs = torch.randn((32, 2, 5, 25, 8, 128), generator=generator, device='cuda:0')
        truth = torch.randn((32, 2, 2, 25, 8, 128), generator=generator, device='cuda:0')
        model = module.LeeAlternativeMeta().to('cuda:0')
        sft_before = branch_tensor_hash(model)
        torch.cuda.reset_peak_memory_stats()
        records = {}
        for phase in ('add_meta', 'joint_online'):
            optimizer = adam(torch, model) if phase == 'joint_online' else None
            torch.cuda.synchronize()
            started = time.perf_counter()
            losses = []
            for _ in range(2):
                check_deadline()
                if phase == 'add_meta':
                    metrics = updates.add_meta_step(model, inputs[:16], truth[:16], inputs[16:], truth[16:],
                                                   microbatch=args.microbatch, deadline_check=check_deadline)
                    losses.append(metrics['query_loss'])
                else:
                    losses.append(online_step(torch, module, model, optimizer, inputs, truth, args.microbatch))
            torch.cuda.synchronize()
            records[phase] = {'actual_updates': 2, 'seconds': time.perf_counter()-started,
                              'finite_losses': losses}
            if phase == 'add_meta' and branch_tensor_hash(model) != sft_before:
                raise ValueError('Probe offline stage changed SFT weights')
        check_deadline()
        peak = int(torch.cuda.max_memory_reserved())
        if peak > 6*2**30:
            raise ValueError('Probe exceeded6GiB Torch allocation budget')
        result.update(status='passed', phases=records,
                      offline_sft_unchanged=True, torch_peak_reserved_bytes=peak,
                      torch_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                      expected_exclusive_seconds_per12000_frames={p: r['seconds']*375/2 for p,r in records.items()})
    except Exception:
        result['failure_traceback'] = traceback.format_exc()
        raise
    finally:
        result['observed_end_utc'] = utc_now().isoformat()
        write_json(output, sign(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'project', 'parent-receipt', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--microbatch', type=int, choices=(1, 2, 4, 8), required=True)
    print(json.dumps(probe(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
