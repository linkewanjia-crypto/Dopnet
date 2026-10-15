"""Resource selection fault checks using tiny administrative JSON only."""
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lee_meta_equal40.artifacts import sign,write_json
from lee_meta_equal40.queue import select_candidate


def valid(micro,meta,online):
    return dict(status='passed',microbatch=micro,synthetic_hardware_only=True,
                scientific_data_generated=False,training_started=False,
                offline_sft_unchanged=True,torch_peak_reserved_bytes=2**30,
                expected_exclusive_seconds_per12000_frames=dict(add_meta=meta,joint_online=online))


class QueueSelectionTests(unittest.TestCase):
    def test_choose_actual40_workload_not_single_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a.json';b=root/'b.json'
            write_json(a,sign(valid(1,1,3)));write_json(b,sign(valid(2,20,1)))
            result=select_candidate([(1,root/'manifest1',a),(2,root/'manifest2',b)])
            self.assertEqual(result[:2],(59,2))

    def test_failed_candidate_is_retained_but_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a.json';b=root/'b.json'
            v=valid(1,1,1);v['status']='failed';write_json(a,sign(v))
            write_json(b,sign(valid(2,2,2)))
            self.assertEqual(select_candidate([(1,root,a),(2,root,b)])[1],2)
            with self.assertRaises(ValueError):select_candidate([(1,root,a)])

    def test_bad_passed_resource_proof_does_not_qualify(self):
        bad=[('microbatch',True),('torch_peak_reserved_bytes',6*2**30+1),
             ('scientific_data_generated',True),('offline_sft_unchanged',False),
             ('training_started',True),('expected_exclusive_seconds_per12000_frames',
              dict(add_meta=1,joint_online=float('nan')))]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a.json'
            for key,value in bad:
                v=valid(1,1,1);v[key]=value
                # NaN cannot enter signed canonical JSON, which is itself a gate.
                if key=='expected_exclusive_seconds_per12000_frames':
                    with self.assertRaises(ValueError):sign(v)
                    continue
                write_json(a,sign(v))
                with self.assertRaises(ValueError):select_candidate([(1,root,a)])


if __name__=='__main__':unittest.main()
