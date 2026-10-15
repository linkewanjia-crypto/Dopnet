"""Allocator-policy regression after actual first waveform OOM; no Torch."""
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lee_meta_equal40.train import process_memory_limit
class ResourcePolicyTests(unittest.TestCase):
    def test_waveform_uses_original_gpu_headroom_probe_keeps6gib(self):
        total=50894602240
        self.assertEqual(process_memory_limit(total),total-3*2**30)
        self.assertEqual(process_memory_limit(total,True),6*2**30)
        for bad in (True,6*2**30,50894602240.0):
            with self.assertRaises(ValueError):process_memory_limit(bad)
        with self.assertRaises(ValueError):process_memory_limit(total,1)
if __name__=='__main__':unittest.main()
