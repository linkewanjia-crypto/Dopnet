"""The actual near-cancelled endpoint failure, using scalar toy metadata only."""
import csv
import math
from pathlib import Path
import tempfile
import unittest
from .verify_wimae import check_seed_intervals


class CriticalValuePatchTests(unittest.TestCase):
    def fixture(self, path, t_value=4.302652729749462):
        a=0.0008
        mean=t_value*a/math.sqrt(3)+1e-13
        values=[mean-a,mean,mean+a]
        sd=math.sqrt(sum((v-mean)**2 for v in values)/2)
        half=t_value*sd/math.sqrt(3)
        row=dict(metric='nmse',n_training_seeds=3,mean=mean,seed_sd=sd,
                 seed42=values[0],seed43=values[1],seed44=values[2],
                 mean_db=10*math.log10(mean),seed_ci95_lo_unbounded=mean-half,
                 seed_ci95_hi_unbounded=mean+half)
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        return row

    def test_legacy_constant_reproduces_nearzero_endpoint_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.csv';self.fixture(path)
            with self.assertRaises(RuntimeError):check_seed_intervals(path,4.302652729696142)
            check_seed_intervals(path,4.302652729749462)

    def test_wrong_endpoint_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.csv';row=self.fixture(path)
            row['seed_ci95_lo_unbounded']+=1e-8
            with path.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            with self.assertRaises(RuntimeError):check_seed_intervals(path,4.302652729749462)

    def test_signed_actual_df2_value_matches_analytic_quantile(self):
        exact=math.sqrt(2*.95**2/(1-.95**2))
        self.assertAlmostEqual(exact,4.302652729749462,places=13)


if __name__=='__main__':unittest.main()
