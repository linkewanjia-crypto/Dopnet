# HISTORICAL_SANITIZED_NON_RUNNABLE: original bytes retained in source archive.
"""Small scalar/metadata fault fixtures only; never import tensor frameworks."""
import copy
import csv
import math
from pathlib import Path
import sys
import tempfile
import unittest
from .protocol import (LABELS, NEW_METHOD, READER_SHA, SEEDS, digest,
                       require_routes, require_training_receipt, safe_name)
from .statistics_checks import check_seed_intervals, check_paired_intervals
from .execution import (MODE, LEGACY_DEADLINE, TRAINING_MANIFEST_SHA, TASKS,
                        check_queue_ready, control_fields, check_receipt_members, check_migration_copies, ExecutionStopped)
from . import execution as X
from unittest.mock import patch
from types import SimpleNamespace


def csv_write(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def fixtures():
    summaries=[];pairs=[];t=math.sqrt(2*.95**2/(1-.95**2))
    for i,method in enumerate(LABELS):
        values=[.10+i*.005,.15+i*.005,.20+i*.005]
        mean=sum(values)/3;sd=math.sqrt(sum((v-mean)**2 for v in values)/2);half=t*sd/math.sqrt(3)
        summaries.append(dict(cell_id='toy_metadata_fixture',metric='ber_diagonal',method=method,n_training_seeds='3',
            seed42=values[0],seed43=values[1],seed44=values[2],mean=mean,seed_sd=sd,seed_ci95_lo_unbounded=mean-half,seed_ci95_hi_unbounded=mean+half))
        if i:
            delta=i*.005
            pairs.append(dict(cell_id='toy_metadata_fixture',metric='ber_diagonal',comparator=method,reference='dopnet',
                paired_difference=delta,paired_seed_ci95_lo=delta,paired_seed_ci95_hi=delta,
                paired_test_cluster_ci95_lo=delta-.001,paired_test_cluster_ci95_hi=delta+.001))
    return summaries,pairs


class MetadataFaultTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.summaries,self.pairs=fixtures()
    def tearDown(self):
        self.temp.cleanup()
    def files(self):
        s=self.root/'summary.csv';p=self.root/'pairs.csv';csv_write(s,self.summaries);csv_write(p,self.pairs);return s,p
    def test_complete8_7_fixture(self):
        s,p=self.files();check_seed_intervals(s);check_paired_intervals(s,p)
    def test_missing_new_Lee_method(self):
        self.summaries.pop();self.pairs.pop();s,p=self.files()
        with self.assertRaises(ValueError):check_paired_intervals(s,p)
    def test_duplicate_summary_group(self):
        self.summaries.append(copy.deepcopy(self.summaries[0]));s,p=self.files()
        with self.assertRaises(ValueError):check_paired_intervals(s,p)
    def test_missing_comparator_pair(self):
        self.pairs.pop();s,p=self.files()
        with self.assertRaises(ValueError):check_paired_intervals(s,p)
    def test_duplicate_pair(self):
        self.pairs[-1]=copy.deepcopy(self.pairs[0]);s,p=self.files()
        with self.assertRaises(ValueError):check_paired_intervals(s,p)
    def test_wrong_reference_pair(self):
        self.pairs[-1]['reference']=NEW_METHOD;s,p=self.files()
        with self.assertRaises(ValueError):check_paired_intervals(s,p)
    def test_wrong_same_seed_delta(self):
        self.pairs[1]['paired_difference']+=.01;s,p=self.files()
        with self.assertRaises(ValueError):check_paired_intervals(s,p)
    def test_bad_cluster_interval(self):
        self.pairs[1]['paired_test_cluster_ci95_lo']=1;s,p=self.files()
        with self.assertRaises(ValueError):check_paired_intervals(s,p)
    def test_df2_endpoint_rounding_regression(self):
        s,_=self.files()
        with self.assertRaises(ValueError):check_seed_intervals(s,4.302)
    def test_nonfinite_seed(self):
        self.summaries[-1]['seed44']='nan';s,_=self.files()
        with self.assertRaises(ValueError):check_seed_intervals(s)
    def test_two_seeds(self):
        self.summaries[-1]['n_training_seeds']='2';s,_=self.files()
        with self.assertRaises(ValueError):check_seed_intervals(s)
    def test_unsafe_zip_paths(self):
        for name in ('../x','/x','x\\y','__HISTORICAL_MACHINE_PATH_REDACTED__','a/../x','a//x'):
            with self.subTest(name=name):self.assertFalse(safe_name(name))
        self.assertTrue(safe_name('training/checkpoints/lee_meta_seed42/commits/epoch40/state.pt'))
    def inherited_routes(self):
        return [dict(method=m,training_seed=s,trained=True) for m in list(LABELS)[:7] for s in SEEDS]
    def test_routes24_exact21_reuse(self):
        inherited=self.inherited_routes();require_routes(inherited+[dict(method=NEW_METHOD,training_seed=s,trained=True) for s in SEEDS],inherited)
    def test_renamed_old_supervised_Lee(self):
        inherited=self.inherited_routes();routes=copy.deepcopy(inherited)+[dict(method=NEW_METHOD,training_seed=s,trained=True) for s in SEEDS]
        routes[6]['method']=NEW_METHOD
        with self.assertRaises(ValueError):require_routes(routes,inherited)
    def test_new_seed_order(self):
        inherited=self.inherited_routes();routes=inherited+[dict(method=NEW_METHOD,training_seed=s,trained=True) for s in reversed(SEEDS)]
        with self.assertRaises(ValueError):require_routes(routes,inherited)
    def execution(self):
        return dict(execution_mode=MODE,execution_control_signature='c'*64,
                    extension_source_signature='e'*64,migration_receipt_sha256='m'*64,
                    legacy_deadline_superseded=True)
    def proof(self):
        from lee_meta_equal40.train import binding_for
        m={'signature':'toy_manifest_signature'};cfg={'lee_meta_reproduction':{'recipe_sha256':'toy_recipe_digest'}}
        p=dict(status='single_seed_training_metadata_pass',exit_code=0,training_delivery_qualified=True,committed_epochs=40,
            binding=binding_for(m,cfg,42),actual_restricted_reader_sha256=READER_SHA,rng_presence_verified=True,rng_boundary_bitwise_verified=False,all_experiments_complete=False,selected_epoch=38,**self.execution())
        return p,m,cfg
    def test_selected_best_before40_valid(self):
        p,m,cfg=self.proof();require_training_receipt(p,42,m,cfg,self.execution())
    def test_bool_exit_code_not_real0(self):
        p,m,cfg=self.proof();p['exit_code']=False
        with self.assertRaises(ValueError):require_training_receipt(p,42,m,cfg,self.execution())
    def test_commit39_does_not_qualify(self):
        p,m,cfg=self.proof();p['committed_epochs']=39
        with self.assertRaises(ValueError):require_training_receipt(p,42,m,cfg,self.execution())
    def test_offline_epoch1_cannot_be_online_best(self):
        p,m,cfg=self.proof();p['selected_epoch']=1
        with self.assertRaises(ValueError):require_training_receipt(p,42,m,cfg,self.execution())
    def test_rng_presence_not_bitwise_proof(self):
        p,m,cfg=self.proof();p['rng_boundary_bitwise_verified']=True
        with self.assertRaises(ValueError):require_training_receipt(p,42,m,cfg,self.execution())
    def test_training_receipt_not_all_experiments_complete(self):
        p,m,cfg=self.proof();p['all_experiments_complete']=True
        with self.assertRaises(ValueError):require_training_receipt(p,42,m,cfg,self.execution())
    def queue(self):
        return dict(schema='lee_execution_extension_queue_v1',status='three_seed_training_metadata_complete_scoring_pending',current=None,
                    tasks={name:dict(status='complete',returncode=0,pid=100+i) for i,name in enumerate(sorted(TASKS))},
                    **self.execution())
    def test_extension_exact6_ready(self):
        check_queue_ready(self.queue(),self.execution())
    def test_extension_missing_metadata_task(self):
        q=self.queue();q['tasks'].pop('verify_seed44')
        with self.assertRaises(ValueError):check_queue_ready(q,self.execution())
    def test_extension_legacy_probe_task_not_public6(self):
        q=self.queue();q['tasks']['probe_micro8']=dict(status='complete',returncode=0,pid=199)
        with self.assertRaises(ValueError):check_queue_ready(q,self.execution())
    def test_extension_boolean_exit_not0(self):
        q=self.queue();q['tasks']['train_seed42']['returncode']=False
        with self.assertRaises(ValueError):check_queue_ready(q,self.execution())
    def test_extension_running_task_not_ready(self):
        q=self.queue();q['tasks']['train_seed43']['status']='running'
        with self.assertRaises(ValueError):check_queue_ready(q,self.execution())
    def test_extension_wrong_control_queue(self):
        q=self.queue();q['execution_control_signature']='wrong'
        with self.assertRaises(ValueError):check_queue_ready(q,self.execution())
    def test_training_receipt_wrong_migration(self):
        p,m,cfg=self.proof();p['migration_receipt_sha256']='wrong'
        with self.assertRaises(ValueError):require_training_receipt(p,42,m,cfg,self.execution())
    def control(self):
        return dict(schema='lee_execution_control_v1',mode=MODE,deadline_utc=None,
                    legacy_deadline_utc=LEGACY_DEADLINE,legacy_source_manifest_sha256=TRAINING_MANIFEST_SHA,
                    signature='c'*64,extension_source_signature='e'*64)
    def test_continuous_policy_no_fake_future_date(self):
        c=self.control();self.assertEqual(control_fields(c,'m'*64),self.execution())
        c['deadline_utc']='2099-01-01T00:00:00Z'
        with self.assertRaises(ValueError):control_fields(c,'m'*64)
    def test_continuous_policy_preserves_legacy_date(self):
        c=self.control();c['legacy_deadline_utc']='2099-01-01T00:00:00Z'
        with self.assertRaises(ValueError):control_fields(c,'m'*64)
    def receipt_fixture(self):
        # Tiny metadata filenames only; no scientific state or arrays are created.
        c=SimpleNamespace(evidence_root=self.root,migration={'prefix_commits':{'42':[{}]*40,'43':[{}]*39,'44':[{}]*40}})
        p=self.root/'execution_commits/lee_meta_seed43/epoch40.json';p.parent.mkdir(parents=True);p.write_text('{}')
        return c,p
    def test_exact_new_commit_receipt_set(self):
        c,p=self.receipt_fixture();check_receipt_members(c)
    def test_missing_new_commit_receipt(self):
        c,p=self.receipt_fixture();p.unlink()
        with self.assertRaises(ValueError):check_receipt_members(c)
    def test_extra_unbound_receipt(self):
        c,p=self.receipt_fixture();p.with_name('epoch39.json').write_text('{}')
        with self.assertRaises(ValueError):check_receipt_members(c)
    def stop_context(self):
        c=self.control();cp=self.root/'EXECUTION_CONTROL.json';mp=self.root/'MIGRATION_RECEIPT.json'
        cp.write_text(__import__('json').dumps(c));mp.write_text('{}')
        return SimpleNamespace(control_path=cp,migration_path=mp,control=c,migration_sha=__import__('hashlib').sha256(mp.read_bytes()).hexdigest())
    def test_exact_user_stop_preserves_with75_exception(self):
        c=self.stop_context()
        stop=dict(schema='lee_execution_user_stop_v1',status='user_requested_stop',execution_control_signature=c.control['signature'],requested_utc='2026-10-02T15:00:00Z')
        stop['signature']=digest(stop);(self.root/'USER_STOP.json').write_text(__import__('json').dumps(stop))
        # A signed arbitrary marker is not evidence of an actual user request;
        # this tiny fixture only tests the authorization-bound sentinel schema.
        c.control['signature']=digest({k:v for k,v in c.control.items() if k!='signature'})
        c.control_path.write_text(__import__('json').dumps(c.control))
        stop['execution_control_signature']=c.control['signature'];stop.pop('signature');stop['signature']=digest(stop)
        (self.root/'USER_STOP.json').write_text(__import__('json').dumps(stop))
        with patch.object(X,'CONTEXT',c),self.assertRaises(ExecutionStopped):X.check_execution()
    def test_foreign_user_stop_fails_closed(self):
        c=self.stop_context();c.control['signature']=digest({k:v for k,v in c.control.items() if k!='signature'})
        c.control_path.write_text(__import__('json').dumps(c.control))
        stop=dict(schema='lee_execution_user_stop_v1',status='user_requested_stop',execution_control_signature='wrong',requested_utc='2026-10-02T15:00:00Z')
        stop['signature']=digest(stop);(self.root/'USER_STOP.json').write_text(__import__('json').dumps(stop))
        with patch.object(X,'CONTEXT',c),self.assertRaises(ValueError):X.check_execution()
    def test_changed_migration_cannot_continue(self):
        c=self.stop_context();c.control['signature']=digest({k:v for k,v in c.control.items() if k!='signature'})
        c.control_path.write_text(__import__('json').dumps(c.control));c.migration_path.write_text('{"changed":1}')
        with patch.object(X,'CONTEXT',c),self.assertRaises(ValueError):X.check_execution()
    def copied_fixture(self):
        training=self.root/'training';training.mkdir();p=training/'lee_manifest.json';p.write_text('{}')
        h=__import__('hashlib').sha256(p.read_bytes()).hexdigest()
        c=SimpleNamespace(control={'extension_source_signature':'e'*64},evidence_root=self.root,
            migration=dict(extension_source_signature='e'*64,legacy_bytes_rewritten=False,scientific_binding_changed=False,all_experiments_complete=False,
                          training_manifest_sha256=h,copied_files={'candidate_micro8/lee_manifest.json':dict(bytes=p.stat().st_size,sha256=h)}))
        return c,training,p
    def test_immutable_migration_copy_hash(self):
        c,t,p=self.copied_fixture();check_migration_copies(c,t)
    def test_mutated_migration_copy_rejected(self):
        c,t,p=self.copied_fixture();p.write_text('{"mutated":true}')
        with self.assertRaises(ValueError):check_migration_copies(c,t)
    def test_missing_legacy_log_copy_rejected(self):
        c,t,p=self.copied_fixture();c.migration['copied_files']['legacy_logs/train_seed42.log']=dict(bytes=1,sha256='x'*64)
        with self.assertRaises(ValueError):check_migration_copies(c,t)
    def test_standalone_archive_clears_deleted_context_pointer(self):
        with patch.object(X,'CONTEXT',SimpleNamespace(control_path=Path('deleted-temp-control'))),patch.object(X,'EXTENSION_ROOT',None):
            X.restore(None,None);self.assertIsNone(X.CONTEXT)
    def test_live_packaging_context_is_restored(self):
        c=SimpleNamespace(control_path=Path('live-control'),migration_path=Path('live-migration'),legacy_root=Path('legacy'),evidence_root=Path('live-evidence'))
        with patch.object(X,'activate') as activate:
            X.restore(c,Path('reviewed-extension'))
            activate.assert_called_once_with(c.control_path,c.migration_path,c.legacy_root,Path('reviewed-extension'),evidence_root=c.evidence_root)
    def test_no_tensor_imports(self):
        self.proof();self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)


if __name__=='__main__':
    unittest.main()
