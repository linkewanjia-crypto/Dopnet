"""Fault-injection checks on metadata/storage; no tensors or scientific data."""
import ast
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lee_meta_equal40.artifacts import (publish_commit, scan_commits, sign,
                                       write_json, derive_summary, check_row)
from lee_meta_equal40.prepare import build_config
from lee_meta_equal40.verify_training import check_state
from lee_meta_equal40.train import DeadlineReached, validate_resume


def row(epoch):
    return dict(global_epoch=epoch, data_epoch=epoch,
                phase='add_meta' if epoch == 1 else 'joint_online',
                fresh_frames=12000, optimizer_updates=375,
                support_frames=6000 if epoch == 1 else 0,
                query_frames=6000 if epoch == 1 else 0,
                meta_inner_steps=375 if epoch == 1 else 0,
                online_lr=None if epoch == 1 else 0.001,
                homogeneous_meta_episodes=375 if epoch == 1 else 0,
                train_loss=1.0, val_mse=1.0, val_nmse_db=0.0, train_seconds=1.0)


def inert_save(state, path):
    # Deliberately inert bytes for exercising file transaction boundaries.
    Path(path).write_bytes(json.dumps(state, sort_keys=True).encode())


class IntegrationFaultTests(unittest.TestCase):
    def test_crash_before_publish_is_not_a_committed_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = dict(epoch=1, history=[row(1)], meta_parent_sha256=None)
            def abort():
                raise DeadlineReached('injected original deadline')
            with self.assertRaises(DeadlineReached):
                publish_commit(tmp, state, {}, inert_save, abort)
            self.assertEqual(scan_commits(tmp, {}), [])
            self.assertEqual(len(list((Path(tmp)/'commits').iterdir())), 1)
            published = publish_commit(tmp, state, {}, inert_save, lambda: None)
            self.assertEqual(scan_commits(tmp, {}), [published])
            with self.assertRaises(FileExistsError):
                publish_commit(tmp, state, {}, inert_save, lambda: None)

    def test_bad_hash_gap_or_binding_does_not_resume(self):
        for mutation in ('state_bytes', 'gap', 'binding'):
            with tempfile.TemporaryDirectory() as tmp:
                publish_commit(tmp, dict(epoch=1, history=[row(1)], meta_parent_sha256=None),
                               {'seed':42}, inert_save, lambda:None)
                p = Path(tmp)/'commits/epoch01'
                if mutation == 'state_bytes':
                    (p/'state.pt').write_bytes(b'changed')
                elif mutation == 'gap':
                    p.rename(p.with_name('epoch02'))
                with self.assertRaises(ValueError):
                    scan_commits(tmp, {'seed':43} if mutation == 'binding' else {'seed':42})

    def test_parent_chain_and_earliest_minimum_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = publish_commit(tmp, dict(epoch=1, history=[row(1)], meta_parent_sha256=None),
                                 {}, inert_save, lambda:None)
            history = [row(1),row(2)]
            publish_commit(tmp, dict(epoch=2, history=history, meta_parent_sha256=one['state_sha256']),
                           {}, inert_save, lambda:None)
            history.append(row(3))
            publish_commit(tmp, dict(epoch=3, history=history, meta_parent_sha256=one['state_sha256']),
                           {}, inert_save, lambda:None)
            summary=derive_summary(scan_commits(tmp,{}))
            self.assertEqual(summary['selected_epoch'],2)
            marker=Path(tmp)/'commits/epoch03/commit.json'
            bad=json.loads(marker.read_text());bad.pop('signature');bad['meta_parent_sha256']='a'*64
            write_json(marker,sign(bad))
            with self.assertRaises(ValueError): scan_commits(tmp,{})

    def test_bad_budget_phase_or_nonfinite_selection_rejected(self):
        for key,value in [('fresh_frames',11999),('optimizer_updates',374),('data_epoch',1),
                          ('global_epoch',True),('val_mse',float('nan')),('online_lr',0.0001)]:
            r=row(2);r[key]=value
            with self.assertRaises(ValueError): check_row(r,2)

    def test_freeze_geometry_and_require_true_ints(self):
        cfg={'system':dict(n_users=2,n_ant=8,n_subcarriers=128,frame_num_symbols=25,
                           threegpp_link_batch_size=8),'data_pipeline':dict(generation_batch_size=32,shuffle_buffer=False),
             'train':dict(train_samples=12000,val_samples=2000,validation_seed=610000001)}
        adapted=build_config(cfg,42,1)
        self.assertEqual(cfg['system'],adapted['system'])
        self.assertEqual(cfg['data_pipeline'],adapted['data_pipeline'])
        for seed,micro in [(True,1),(42,True),(42,16)]:
            with self.assertRaises(ValueError):build_config(cfg,seed,micro)
        bad=copy.deepcopy(cfg);bad['system']['threegpp_link_batch_size']=4
        with self.assertRaises(ValueError):build_config(bad,42,1)

    def test_ast_explicit_recompute_parameters_preserve_inner_chain(self):
        tree=ast.parse((ROOT/'lee_meta_equal40/meta_updates.py').read_text())
        checkpoints=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='checkpoint']
        self.assertEqual(len(checkpoints),1)
        self.assertIs(next(k.value.value for k in checkpoints[0].keywords if k.arg=='use_reentrant'),False)
        self.assertIn('*tuple(parameters.values())',ast.unparse(checkpoints[0]))
        function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='loss_function')
        self.assertIn('OrderedDict(zip(names, values))',ast.unparse(function))
        self.assertEqual(len(function.args.kwonlyargs),2) # freezes each micro's x/h during recomputation

    def test_metadata_import_graph_is_torch_free(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('lee_meta_equal40.model',sys.modules)

    def test_training_grad_context_does_not_inherit_generator_no_grad(self):
        tree=ast.parse((ROOT/'lee_meta_equal40/train.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_epoch')
        enabled=[n for n in ast.walk(function) if isinstance(n,ast.With) and
                 any('torch.enable_grad()'==ast.unparse(i.context_expr) for i in n.items)]
        self.assertEqual(len(enabled),2)
        self.assertIn('updates.add_meta_step',ast.unparse(enabled[0]))
        self.assertIn('online_step',ast.unparse(enabled[1]))

    def test_restore_rejects_environment_or_phase_mismatch(self):
        r=row(1);record={'history_sha256':'unused','meta_parent_sha256':None}
        base={'binding':{},'cfg':{},'environment':{},'epoch':1,'history':[r],'model':{},
              'optimizer':None,'meta_parent_sha256':None,'rng_python':(),'rng_numpy':(),
              'rng_torch':'inert','rng_cuda':['inert']}
        from lee_meta_equal40.artifacts import digest
        record['history_sha256']=digest([r])
        validate_resume(base,record,[r],{}, {}, {})
        for key,value in [('environment',{'torch':'different'}),('optimizer',{}),('epoch',2),('rng_cuda',None)]:
            bad=dict(base);bad[key]=value
            with self.assertRaises(ValueError):validate_resume(bad,record,[r],{},{},{})

    def test_restricted_state_metadata_requires_both_branches_and_full_adam_rng(self):
        from lee_meta_equal40.artifacts import digest
        weights={}
        for branch in ('sft','add'):
            for layer in ('feature','shrink','mapping.0','mapping.1','mapping.2','mapping.3','expand','reconstruct'):
                for kind in ('weight','bias'):weights[f'{branch}.{layer}.{kind}']='inert_tensor'
        history=[row(1),row(2)]; binding={'seed':42}; cfg={}; parent='a'*64
        state=dict(schema='lee_equal40_full_state_v1',binding=binding,cfg=cfg,epoch=2,data_epoch=2,
                   phase='joint_online',history=history,meta_parent_sha256=parent,model=weights,
                   rng_python=(3,(),None),rng_numpy=('MT19937','inert',1,0,0.0),
                   rng_torch='inert',rng_cuda=['inert'],initial_sft_sha256='b'*64,
                   generator_identity=dict(seed=10043,last_global_epoch=2,next_global_epoch=3,
                                           generation_batch=32,link_batch=8,stateless_per_epoch=True),
                   optimizer_name='Adam',adam_step_counts=[375]*32,
                   optimizer={'param_groups':[dict(lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,params=list(range(32)))],
                              'state':{i:dict(step='inert',exp_avg='inert',exp_avg_sq='inert') for i in range(32)}})
        record={'epoch':2,'history_sha256':digest(history)}
        check_state(state,record,history,binding,cfg,parent)
        for key,value in [('rng_cuda',[]),('meta_parent_sha256','c'*64),('adam_step_counts',[374]*32),
                          ('model',{'add.feature.weight':'inert'}),('optimizer',{})]:
            bad=dict(state);bad[key]=value
            with self.assertRaises(ValueError):check_state(bad,record,history,binding,cfg,parent)


if __name__ == '__main__':
    unittest.main()
