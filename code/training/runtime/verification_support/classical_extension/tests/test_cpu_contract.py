"""Small file-contract fixtures, not channel simulations or research results."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_classical_cpu as cpu


class CPUContractTests(unittest.TestCase):
    def test_calibration_choice_is_create_once(self):
        with tempfile.TemporaryDirectory(prefix='dopnet_cpu_contract_') as directory:
            path = Path(directory) / 'calibration_lock.json'
            cpu.create_json_once(path, {'fixture': 'first choice'})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                cpu.create_json_once(path, {'fixture': 'different choice'})
            self.assertEqual(path.read_bytes(), before)

    def test_output_has_one_writer_and_lock_releases(self):
        with tempfile.TemporaryDirectory(prefix='dopnet_cpu_contract_') as directory:
            root = Path(directory)
            with cpu.output_writer_lock(root):
                with self.assertRaises(RuntimeError):
                    with cpu.output_writer_lock(root):
                        self.fail('Second writer entered')
            with cpu.output_writer_lock(root):
                pass

    def test_wrong_independent_calibration_seed_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='dopnet_cpu_contract_') as directory:
            root = Path(directory)
            (root / 'calibration').mkdir()
            raw = root / 'calibration' / 'toy_contract.bin'
            raw.write_bytes(b'Toy contract fixture, never receiver data')
            manifest = {'export_signature': 'fixture', 'calibration_seed': 810000001}
            marker = {'schema_version': 1, 'export_signature': 'fixture',
                      'role': 'calibration', 'samples': 128, 'seed': 810000001,
                      'batches': [{'path': 'calibration/toy_contract.bin', 'offset': 0,
                                   'samples': 128, 'sha256': cpu.file_hash(raw)}]}
            path = root / 'calibration' / 'COMPLETE.json'
            cpu.atomic_json(path, marker)
            self.assertEqual(cpu.read_complete(root, manifest, 'calibration')[0]['samples'], 128)
            marker['seed'] = 42
            cpu.atomic_json(path, marker)
            with self.assertRaises(RuntimeError):
                cpu.read_complete(root, manifest, 'calibration')


if __name__ == '__main__':
    unittest.main()
