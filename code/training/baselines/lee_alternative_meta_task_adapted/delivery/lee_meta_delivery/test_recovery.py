"""New pure metadata/AST faults only; no NumPy, actual scores or tensor use."""
import ast
import copy
from pathlib import Path
import sys
import unittest
from . import recovery as R

class RecoveryMetadataTests(unittest.TestCase):
    def test_scope_all33(self):
        for fd in (800,1200,1600):
            for channel in range(11):
                c=dict(ber=True,modulation='16qam',snr_db=20,doppler_hz=fd)
                keys=R.CORE+(R.NOISE if fd==1200 else ())
                self.assertEqual(R.receiver_keys(c,keys),keys)
    def test_scope_faults(self):
        c=dict(ber=True,modulation='16qam',snr_db=20,doppler_hz=1200)
        for key in R.CORE+R.NOISE:
            with self.subTest(key=key),self.assertRaises(ValueError):R.receiver_keys(c,[k for k in R.CORE+R.NOISE if k!=key])
        for field,value in [('doppler_hz',800),('doppler_hz',1600),('snr_db',10),('modulation','qpsk')]:
            bad=dict(c);bad[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):R.receiver_keys(bad,R.CORE+R.NOISE)
    def test_nmse_no_fictitious_receiver(self):
        self.assertEqual(R.receiver_keys(dict(ber=False),['nmse','signature']),())
        for key in R.CORE+R.NOISE:
            with self.subTest(key=key),self.assertRaises(ValueError):R.receiver_keys(dict(ber=False),['nmse',key])
    def fixtures(self):
        ev=dict(implementation_signature=R.OLD_SOURCE_SIG,signature='a'*64,cells=[dict(cell_id='toy_nmse_%03d'%i,ber=False) for i in range(236)]+[dict(cell_id='toy_ber_%03d'%i,ber=True) for i in range(33)],routes=[dict(method='toy',training_seed=42)]*24)
        records={c['cell_id']:dict(score_sha256='b'*64,marker_sha256='c'*64) for c in ev['cells'] if not c['ber']}
        evidence={n:'d'*64 for n in ('evaluation.json','training_queue_state.json','control/queue_state.json','control/evaluate.log','launch/launch_receipt.json','launch/launch_request.json')}
        evidence['source/SOURCE_MANIFEST.json']=R.OLD_SOURCE_SHA
        for cid,row in records.items():
            evidence['cells/'+cid+'.npz']=row['score_sha256'];evidence['cells/'+cid+'.COMPLETE.json']=row['marker_sha256']
        return ev,records,evidence
    def test_binding_complete(self):
        ev,r,e=self.fixtures();b=R.reuse_binding(ev,r,e)
        self.assertEqual(b['reused_NMSEonly_cells'],236);self.assertEqual(b['reused_BER_cells'],0)
    def test_binding_faults(self):
        ev,r,e=self.fixtures()
        mutations=[]
        v=copy.deepcopy(r);v.pop(next(iter(v)));mutations.append((ev,v,e))
        v=copy.deepcopy(r);v['toy_ber_000']=v.pop('toy_nmse_000');mutations.append((ev,v,e))
        v=copy.deepcopy(r);v['toy_nmse_000']['score_sha256']='bad';mutations.append((ev,v,e))
        v=copy.deepcopy(ev);v['implementation_signature']='wrong';mutations.append((v,r,e))
        v=copy.deepcopy(e);v['source/SOURCE_MANIFEST.json']='e'*64;mutations.append((ev,r,v))
        v=copy.deepcopy(e);v.pop('control/evaluate.log');mutations.append((ev,r,v))
        v=copy.deepcopy(e);v['cells/toy_nmse_000.npz']='e'*64;mutations.append((ev,r,v))
        v=copy.deepcopy(e);v['../escape']='e'*64;mutations.append((ev,r,v))
        for args in mutations:
            with self.subTest(args=args[0]['signature']),self.assertRaises(ValueError):R.reuse_binding(*args)
    def test_evaluation_only_source_and_recovery_change(self):
        old,r,e=self.fixtures();new=copy.deepcopy(old);new.update(signature='f'*64,implementation_signature='frozen_new_source',scoring_recovery=R.reuse_binding(old,r,e));R.match_evaluations(old,new)
        for key,value in [('routes',[]),('samples_per_cell',1),('test_adaptation_steps',1)]:
            bad=copy.deepcopy(new);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):R.match_evaluations(old,bad)
        bad=copy.deepcopy(new);bad['scoring_recovery']['parent_evaluation_signature']='g'*64
        with self.assertRaises(ValueError):R.match_evaluations(old,bad)
    def test_exact_failure_identity(self):
        q=dict(schema='lee_extension_scoring_queue_v1',source_signature=R.OLD_SOURCE_SIG,status='failed_or_incomplete',current='evaluate',exit_code=1,reason='evaluate failed; retain all evidence',all_experiments_complete=False,tasks=dict(evaluate=dict(status='incomplete',returncode=1,pid=169804)))
        R.failed_queue(q)
        for key,value in [('current','report'),('exit_code',True),('status','running'),('source_signature','wrong'),('all_experiments_complete',True)]:
            bad=copy.deepcopy(q);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):R.failed_queue(bad)
        for key,value in [('returncode',False),('pid',1),('status','complete')]:
            bad=copy.deepcopy(q);bad['tasks']['evaluate'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):R.failed_queue(bad)
    def test_recovery_request_lineage(self):
        request=dict(schema='lee_scoring_environment_recovery_request_v1',administrative_source_sha256=R.REC_ADMIN_SHA,preconditions=dict(signature='b'*64,failure_parent={'original':'a'*64}))
        receipt=dict(schema='lee_scoring_environment_recovery_launch_receipt_v1',pid=169795,administrative_source_sha256=R.REC_ADMIN_SHA,launched_utc='2026-10-02T23:27:30.778973+00:00',launch_request_sha256='c'*64,preconditions_signature='b'*64,failure_parent={'original':'a'*64},expected_output=str(R.PARENT),expected_queue_control=str(R.CONTROL))
        R.recovery_launch(receipt,request,'c'*64)
        for key,value in [('pid',True),('pid',156301),('launched_utc','wrong'),('launch_request_sha256','d'*64),('expected_output','wrong'),('expected_queue_control','wrong'),('preconditions_signature','e'*64),('failure_parent',{})]:
            bad=dict(receipt);bad[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):R.recovery_launch(bad,request,'c'*64)
    def test_scoring_AST_has_no_changed_waveform_or_optimizer(self):
        p=Path(__file__).resolve().parent
        src=(p/'evaluate.py').read_text(encoding='utf-8');tree=ast.parse(src)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertTrue(any(isinstance(n.func,ast.Attribute) and n.func.attr=='reuse_completed' for n in calls))
        self.assertTrue(any(isinstance(n.func,ast.Attribute) and n.func.attr=='receiver_keys' for n in calls))
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step') for n in calls))
    def test_no_framework_import(self):
        self.assertFalse(any(n.split('.')[0] in {'torch','numpy','tensorflow','sionna','jax'} for n in sys.modules))

if __name__=='__main__':unittest.main(verbosity=2)
