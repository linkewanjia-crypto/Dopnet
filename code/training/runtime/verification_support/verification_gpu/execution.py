"""Select the original GPU waveform generator without CPU/GPU speed experiments."""
from pathlib import Path
from verification_extension import protocol as P


def select_generation_backend(args):
    _, experiment, _ = P.check_reference(args.reference_results)
    policy = {'complete': True, 'selected_device': 'cuda:0', 'mode': 'strict_frozen_replay',
              'cpu_gpu_benchmark_performed': False,
              'selection_basis': 'Use the existing CUDA/Sionna batched generator and original frozen random-stream contract',
              'reference_experiment_signature': experiment['signature'], 'suite': args.suite,
              'adapter_sha256': P.file_hash(__file__)}
    P.write_json(Path(args.output) / 'generation_execution.json', policy)
    print('Waveform generation: cuda:0. No CPU/GPU speed benchmark; original full-cell waveform checks retained.', flush=True)
    return policy
