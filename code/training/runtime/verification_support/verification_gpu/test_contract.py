"""Small metadata/reporting checks; no neural inference or waveform generation."""
import json
from pathlib import Path
import tempfile
import unittest

from verification_gpu.reporting import gpu_latency_assets
from resume_gpu_verification import parser


class GPUContracts(unittest.TestCase):
    def test_resume_has_no_generation_or_cpu_benchmark_stage(self):
        args = parser().parse_args(['--reference-results', 'unused'])
        self.assertEqual(args.stage, 'resume')
        self.assertEqual(args.from_stage, 'latency-gpu')
        with self.assertRaises(SystemExit):
            parser().parse_args(['--reference-results', 'unused', '--stage', 'generation-benchmark'])

    def fixture(self, root, missing_seed=False, cpu=False):
        output = root / 'latency_gpu'; output.mkdir()
        rows = []
        for batch in (1, 8):
            for seed in ((42, 43) if missing_seed else (42, 43, 44)):
                rows.append({'method': 'dopnet', 'seed': seed, 'batch_size': batch,
                    'device': 'cpu' if cpu else 'cuda:0', 'source_name': 'calibration',
                    'input_boundary': 'received_occupied_ofdm_grid', 'receiver_mode': 'common_diagonal',
                    'status': 'measured', 'numerical_equivalence': {'passed': True},
                    'timing': {'host_origin_total': {'latency_ms_p50': 2., 'latency_ms_p95': 3.,
                                                    'frames_per_second': 500. * batch}}})
        (output / 'digital_receiver_latency.json').write_text(json.dumps({'complete': True,
            'protocol': {'experiment_signature': 'toy', 'devices': ['cuda:0'],
                         'seeds': [42, 43, 44], 'methods': ['dopnet']},
            'rows': rows, 'hardware': {}, 'device_recommendations': []}), encoding='utf-8')
        return output

    def test_gpu_only_tables_and_complete_three_seeds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); dest = root / 'report'; dest.mkdir()
            files, status = gpu_latency_assets(dest, self.fixture(root), 'toy')
            self.assertEqual(status['status'], 'complete')
            for name in ('corrected_digital_receiver_latency.tex', 'corrected_digital_receiver_throughput.tex'):
                self.assertIn(name, files)
                text = (dest / name).read_text(encoding='utf-8')
                self.assertIn('GPU', text)
                self.assertNotIn('CPU', text)
            self.assertIn('2.000 / 3.000', (dest / 'corrected_digital_receiver_latency.tex').read_text())

    def test_missing_seed_is_not_published_as_complete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); dest = root / 'report'; dest.mkdir()
            files, status = gpu_latency_assets(dest, self.fixture(root, missing_seed=True), 'toy')
            self.assertEqual(status['status'], 'incomplete_gpu_workloads')
            self.assertNotIn('corrected_digital_receiver_latency.tex', files)

    def test_cpu_timing_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); dest = root / 'report'; dest.mkdir()
            with self.assertRaisesRegex(ValueError, 'CPU rows'):
                gpu_latency_assets(dest, self.fixture(root, cpu=True), 'toy')


if __name__ == '__main__':
    unittest.main()
