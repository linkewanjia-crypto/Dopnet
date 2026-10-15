import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from aggregate_results import aggregate, seed_interval, summarize
from experiment_protocol import digest, file_hash

class StatisticalContract(unittest.TestCase):
    def test_small_sample_t(self):
        mean,sd,lo,hi=seed_interval([1,2,3])
        self.assertAlmostEqual(mean,2)
        self.assertAlmostEqual(sd,1)
        self.assertAlmostEqual(hi-mean,4.302652729911275/np.sqrt(3))
    def test_incomplete_seeds_rejected(self):
        with self.assertRaises(ValueError): seed_interval([1,2])
    def test_paired_cancellation(self):
        routes=[dict(method=m,training_seed=s,trained=True) for m in ('dopnet','control') for s in (42,43,44)]
        point=np.array([.1,.2,.3,.11,.21,.31])
        boot=np.tile(point,(2000,1))
        rows,diffs=summarize(point,boot,routes,{},'nmse')
        self.assertAlmostEqual(diffs[0]['paired_difference'],.01)
        self.assertAlmostEqual(diffs[0]['paired_seed_ci95_hi'],.01)
        self.assertAlmostEqual(rows[0]['mean_db'],10*np.log10(.2))
        self.assertNotAlmostEqual(rows[0]['mean_db'],np.mean(10*np.log10(point[:3])))
    def test_duplicate_seed_rejected(self):
        routes=[dict(method='dopnet',training_seed=s,trained=True) for s in (42,42,44)]
        with self.assertRaises(ValueError): summarize(np.ones(3),np.ones((1000,3)),routes,{},'nmse')
    def test_statistics_binds_raw_cell_and_every_summary_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            (out/'cells').mkdir()
            routes=[dict(method='dopnet',training_seed=s,trained=True) for s in (42,43,44)]
            meta=dict(cell_id='fixture',channel='awgn',doppler_hz=0.0,snr_db=15.0,
                      modulation='qpsk',studies=['nmse'],ber=False)
            ev=dict(signature='fixture-evaluation',routes=routes,cells=[meta])
            (out/'evaluation.json').write_text(json.dumps(ev),encoding='utf-8')
            (out/'evaluation_complete.json').write_text(json.dumps(
                dict(signature=ev['signature'],cells=1)),encoding='utf-8')
            raw=out/'cells'/'fixture.npz'
            np.savez_compressed(raw,signature=np.array(ev['signature']),
                                meta=np.array(json.dumps(meta)),
                                cluster_id=np.arange(64)//2,
                                nmse=np.full((3,64),.1))
            aggregate(out,1000)
            stats=json.loads((out/'summary'/'statistics.json').read_text(encoding='utf-8'))
            self.assertEqual(stats['data_binding_version'],1)
            self.assertEqual(stats['cell_file_sha256'],{'fixture':file_hash(raw)})
            self.assertEqual(set(stats['summary_csv_sha256']),{
                'cell_summary.csv','cell_paired_differences.csv','macro_summary.csv',
                'macro_paired_differences.csv','receiver_diagnostics.csv'})
            for name,sha in stats['summary_csv_sha256'].items():
                self.assertEqual(sha,file_hash(out/'summary'/name))
            self.assertEqual(stats['signature'],digest({k:v for k,v in stats.items()
                                                       if k!='signature'}))
if __name__=='__main__': unittest.main()
