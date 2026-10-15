# HISTORICAL_SANITIZED_NON_RUNNABLE: original bytes retained in source archive.
"""Standard-library toy metadata/fault tests; never run model/train/source payload."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from .control import (LEGACY_DEADLINE, LEGACY_SOURCE_SHA, Context, check_control,
                      check_migration, digest, file_hash, sign, signed, stop_requested,
                      timestamp, write)


def control_fixture():
    return sign(dict(schema='lee_execution_control_v1',mode='until_complete_or_explicit_user_stop',deadline_utc=None,legacy_deadline_utc=LEGACY_DEADLINE,legacy_source_manifest_sha256=LEGACY_SOURCE_SHA,extension_source_signature='a'*64,authorization_text='Run until complete or I explicitly stop it',authorization_utc='2026-10-02T12:30:00+00:00',authorization_time_basis='agent_recorded_after_actual_user_message',created_utc='2026-10-02T12:31:00+00:00',legacy_code_root='__HISTORICAL_MACHINE_PATH_REDACTED__',legacy_results='__HISTORICAL_MACHINE_PATH_REDACTED__',legacy_launch='__HISTORICAL_MACHINE_PATH_REDACTED__',new_output='__HISTORICAL_MACHINE_PATH_REDACTED__',project='__HISTORICAL_MACHINE_PATH_REDACTED__',parent_receipt='__HISTORICAL_MACHINE_PATH_REDACTED__',forti_code_root='__HISTORICAL_MACHINE_PATH_REDACTED__',purchase_or_renew_authorized=False,all_experiments_complete=False))


def migration_fixture(c):
    return sign(dict(schema='lee_execution_migration_v1',status='immutable_prefix_copied_and_verified',execution_control_signature=c['signature'],legacy_queue_status='incomplete_deadline',legacy_queue_exit_code=75,prefix_commits={'42':[dict(epoch=1,state_sha256='b'*64,state_bytes=5,commit_sha256='c'*64,submitted_utc='2026-10-02T13:00:00+00:00')],'43':[],'44':[]},copied_utc='2026-10-02T13:47:00+00:00'))


class MetadataTests(unittest.TestCase):
    def test_valid_open_ended(self):
        self.assertIsNone(check_control(control_fixture())['deadline_utc'])
    def test_wrong_schema(self):
        c=control_fixture();c['schema']='unknown'
        with self.assertRaises(ValueError):check_control(c)
    def test_fictitious_future_deadline(self):
        c=control_fixture();c['deadline_utc']='9999-01-01T00:00:00Z'
        with self.assertRaises(ValueError):check_control(c)
    def test_missing_user_authorization(self):
        c=control_fixture();c['authorization_text']=''
        with self.assertRaises(ValueError):check_control(c)
    def test_no_purchase(self):
        c=control_fixture();c['purchase_or_renew_authorized']=True
        with self.assertRaises(ValueError):check_control(c)
    def test_no_overall_complete(self):
        c=control_fixture();c['all_experiments_complete']=True
        with self.assertRaises(ValueError):check_control(c)
    def test_wrong_legacy_source(self):
        c=control_fixture();c['legacy_source_manifest_sha256']='b'*64
        with self.assertRaises(ValueError):check_control(c)
    def test_server_posix_path_portable(self):
        self.assertEqual(check_control(control_fixture())['project'],'__HISTORICAL_MACHINE_PATH_REDACTED__')
    def test_relative_server_path(self):
        c=control_fixture();c['project']='relative'
        with self.assertRaises(ValueError):check_control(c)
    def test_parent_traversal(self):
        c=control_fixture();c['project']='__HISTORICAL_MACHINE_PATH_REDACTED__'
        with self.assertRaises(ValueError):check_control(c)
    def test_naive_timestamp(self):
        with self.assertRaises(ValueError):timestamp('2026-10-02T12:00:00')
    def test_nonutc_timestamp(self):
        with self.assertRaises(ValueError):timestamp('2026-10-02T20:00:00+08:00')
    def test_authorization_after_creation(self):
        c=control_fixture();c['authorization_utc']='2026-10-02T14:00:00Z'
        with self.assertRaises(ValueError):check_control(c)
    def test_migration_valid(self):
        c=control_fixture();self.assertEqual(check_migration(migration_fixture(c),c)['legacy_queue_exit_code'],75)
    def test_migration_rc_bool(self):
        c=control_fixture();m=migration_fixture(c);m['legacy_queue_exit_code']=True
        with self.assertRaises(ValueError):check_migration(m,c)
    def test_migration_non75(self):
        c=control_fixture();m=migration_fixture(c);m['legacy_queue_exit_code']=0
        with self.assertRaises(ValueError):check_migration(m,c)
    def test_prefix_gap(self):
        c=control_fixture();m=migration_fixture(c);m['prefix_commits']['42'][0]['epoch']=2
        with self.assertRaises(ValueError):check_migration(m,c)
    def test_old_after_deadline(self):
        c=control_fixture();m=migration_fixture(c);m['prefix_commits']['42'][0]['submitted_utc']='2026-10-02T13:47:00Z'
        with self.assertRaises(ValueError):check_migration(m,c)
    def test_missing_seed(self):
        c=control_fixture();m=migration_fixture(c);del m['prefix_commits']['44']
        with self.assertRaises(ValueError):check_migration(m,c)
    def test_missing_sha(self):
        c=control_fixture();m=migration_fixture(c);m['prefix_commits']['42'][0]['state_sha256']=''
        with self.assertRaises(ValueError):check_migration(m,c)
    def test_foreign_control_migration(self):
        c=control_fixture();m=migration_fixture(c);m['execution_control_signature']='b'*64
        with self.assertRaises(ValueError):check_migration(m,c)
    def test_full40_skip_supported(self):
        c=control_fixture();m=migration_fixture(c);first=m['prefix_commits']['42'][0];m['prefix_commits']['42']=[dict(first,epoch=i) for i in range(1,41)]
        self.assertEqual(len(check_migration(m,c)['prefix_commits']['42']),40)
    def test_all_finished_no_extension(self):
        c=control_fixture();m=migration_fixture(c);first=m['prefix_commits']['42'][0]
        for seed in ('42','43','44'):m['prefix_commits'][seed]=[dict(first,epoch=i) for i in range(1,41)]
        with self.assertRaises(ValueError):check_migration(m,c)
    def test_signature_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json';c=control_fixture();c['mode']='fake';write(p,c)
            with self.assertRaises(ValueError):signed(p)
    def test_stop_absent(self):
        with tempfile.TemporaryDirectory() as d:self.assertFalse(stop_requested(Path(d)/'EXECUTION_CONTROL.json',control_fixture()))
    def test_stop_signed_exact(self):
        with tempfile.TemporaryDirectory() as d:
            c=control_fixture();write(Path(d)/'USER_STOP.json',sign(dict(schema='lee_execution_user_stop_v1',status='user_requested_stop',execution_control_signature=c['signature'],requested_utc='2026-10-02T14:00:00Z')))
            self.assertTrue(stop_requested(Path(d)/'EXECUTION_CONTROL.json',c))
    def test_stop_wrong_control(self):
        with tempfile.TemporaryDirectory() as d:
            c=control_fixture();write(Path(d)/'USER_STOP.json',sign(dict(schema='lee_execution_user_stop_v1',status='user_requested_stop',execution_control_signature='b'*64,requested_utc='2026-10-02T14:00:00Z')))
            with self.assertRaises(ValueError):stop_requested(Path(d)/'EXECUTION_CONTROL.json',c)
    def test_stop_tampered(self):
        with tempfile.TemporaryDirectory() as d:
            c=control_fixture();write(Path(d)/'USER_STOP.json',dict(status='user_requested_stop',signature='b'*64))
            with self.assertRaises(ValueError):stop_requested(Path(d)/'EXECUTION_CONTROL.json',c)
    def test_copy_exact_metadata_bytes(self):
        from .migrate import copy_exact
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'a.json').write_bytes(b'{"toy_scalar":2}\n');files={};copy_exact(root/'a.json',root/'b.json',files,'b.json')
            self.assertEqual((root/'a.json').read_bytes(),(root/'b.json').read_bytes());self.assertEqual(files['b.json']['sha256'],file_hash(root/'a.json'))
    def test_copy_refuse_existing(self):
        from .migrate import copy_exact
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'a').write_text('toy');(root/'b').write_text('old')
            with self.assertRaises(ValueError):copy_exact(root/'a',root/'b',{},'b')
    def test_no_framework_imported(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)

    def receipt_fixture(self, root):
        ctx=object.__new__(Context);ctx.control=control_fixture();ctx.migration=migration_fixture(ctx.control);ctx.migration_sha='d'*64;ctx.evidence_root=root
        folder=root/'checkpoints/lee_meta_seed42';p=folder/'commits/epoch02';p.mkdir(parents=True)
        record=sign(dict(binding={'seed':42},epoch=2,state_sha256='b'*64,state_bytes=5,submitted_utc='2026-10-02T13:48:00Z'))
        write(p/'commit.json',record)
        receipt=sign(dict(schema='lee_execution_commit_control_v1',status='committed',execution_control_signature=ctx.control['signature'],extension_source_signature=ctx.control['extension_source_signature'],migration_receipt_sha256=ctx.migration_sha,seed=42,epoch=2,state_sha256=record['state_sha256'],state_bytes=5,commit_sha256=file_hash(p/'commit.json'),submitted_utc=record['submitted_utc'],committed_utc='2026-10-02T13:49:00Z'))
        write(ctx.receipt_path(42,2),receipt)
        return ctx,folder,record,receipt

    def test_extension_commit_valid(self):
        with tempfile.TemporaryDirectory() as d:
            ctx,folder,record,_=self.receipt_fixture(Path(d));ctx.check_commit_control(folder,record)
    def test_extension_receipt_source(self):
        with tempfile.TemporaryDirectory() as d:
            ctx,folder,record,r=self.receipt_fixture(Path(d));r['extension_source_signature']='bad';write(ctx.receipt_path(42,2),sign({k:v for k,v in r.items() if k!='signature'}))
            with self.assertRaises(ValueError):ctx.check_commit_control(folder,record)
    def test_extension_receipt_time(self):
        with tempfile.TemporaryDirectory() as d:
            ctx,folder,record,r=self.receipt_fixture(Path(d));r['submitted_utc']='2026-10-02T13:50:00Z';write(ctx.receipt_path(42,2),sign({k:v for k,v in r.items() if k!='signature'}))
            with self.assertRaises(ValueError):ctx.check_commit_control(folder,record)
    def test_extension_receipt_trueint_seed(self):
        with tempfile.TemporaryDirectory() as d:
            ctx,folder,record,r=self.receipt_fixture(Path(d));r['seed']=True;write(ctx.receipt_path(42,2),sign({k:v for k,v in r.items() if k!='signature'}))
            with self.assertRaises(ValueError):ctx.check_commit_control(folder,record)
    def test_extension_receipt_intent_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            ctx,folder,record,r=self.receipt_fixture(Path(d));r['status']='publication_intent';write(ctx.receipt_path(42,2),sign({k:v for k,v in r.items() if k!='signature'}))
            with self.assertRaises(ValueError):ctx.check_commit_control(folder,record)
    def test_extension_receipt_wrong_sha(self):
        with tempfile.TemporaryDirectory() as d:
            ctx,folder,record,r=self.receipt_fixture(Path(d));r['commit_sha256']='e'*64;write(ctx.receipt_path(42,2),sign({k:v for k,v in r.items() if k!='signature'}))
            with self.assertRaises(ValueError):ctx.check_commit_control(folder,record)
    def test_extension_before_migration(self):
        with tempfile.TemporaryDirectory() as d:
            ctx,folder,record,r=self.receipt_fixture(Path(d));ctx.migration['copied_utc']='2026-10-02T14:00:00Z'
            with self.assertRaises(ValueError):ctx.check_commit_control(folder,record)
    def test_missing_extension_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            ctx,folder,record,r=self.receipt_fixture(Path(d));ctx.receipt_path(42,2).unlink()
            with self.assertRaises(FileNotFoundError):ctx.check_commit_control(folder,record)
    def test_prefix_changed_commit(self):
        with tempfile.TemporaryDirectory() as d:
            ctx,folder,record,r=self.receipt_fixture(Path(d));ctx.migration['prefix_commits']['42'].append(dict(epoch=2,state_sha256=record['state_sha256'],state_bytes=5,commit_sha256='e'*64,submitted_utc=record['submitted_utc']))
            with self.assertRaises(ValueError):ctx.check_commit_control(folder,record)
    def test_no_hot_edit_static(self):
        import ast
        tree=ast.parse((Path(__file__).parent/'overlay.py').read_text(encoding='utf-8'))
        self.assertNotIn('torch',[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names])
        text=(Path(__file__).parent/'overlay.py').read_text(encoding='utf-8')
        self.assertIn("kwargs['deadline_check'] = t._extension_context.check",text)
        self.assertIn('t.scan_commits = a.scan_commits = v.scan_commits = controlled_scan',text)
        self.assertIn('t._extension_context = context',text)
    def test_source_member_modified(self):
        from .control import source
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'module.py';p.write_text('VALUE=2\n');m=sign(dict(members={'module.py':dict(bytes=p.stat().st_size,sha256=file_hash(p))}));write(root/'SOURCE_MANIFEST.json',m)
            source(root);p.write_text('VALUE=3\n')
            with self.assertRaises(ValueError):source(root)
    def test_source_manifest_sha_modified(self):
        from .control import source
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write(root/'SOURCE_MANIFEST.json',sign(dict(members={})))
            with self.assertRaises(ValueError):source(root,'b'*64)
    def test_live_worker_gate(self):
        from unittest.mock import patch
        from .control import no_live_training
        with patch('lee_execution_extension.control.process_words',return_value=[(111,['python','-m','lee_execution_extension.worker','--reference-token','lee_meta_equal40.train'])]):
            with self.assertRaises(ValueError):no_live_training()
    def test_only_controller_pid_allowed(self):
        from unittest.mock import patch
        from .control import no_live_training
        with patch('lee_execution_extension.control.process_words',return_value=[(111,['python','-m','lee_execution_extension.controller'])]):
            no_live_training(allowed_pid=111)
            with self.assertRaises(ValueError):no_live_training(allowed_pid=112)
    def test_worker_declares_gpu_empty_gate(self):
        text=(Path(__file__).parent/'worker.py').read_text(encoding='utf-8')
        self.assertIn("require(not gpu.stdout.strip()",text)
        self.assertLess(text.index("require(not gpu.stdout.strip()"),text.index('result = t.train('))
    def test_waiter_no_copy_before_legacy_and_gpu(self):
        text=(Path(__file__).parent/'controller.py').read_text(encoding='utf-8')
        self.assertLess(text.index('legacy_terminal(c)'),text.index('migrated = migrate(cp)'))
        self.assertLess(text.index('if gpu.stdout.strip():'),text.index('migrated = migrate(cp)'))
    def test_child_pythonpath_and_reference_token(self):
        text=(Path(__file__).parent/'controller.py').read_text(encoding='utf-8')
        self.assertIn("env['PYTHONPATH']",text);self.assertIn("'--reference-token'",text)
        self.assertIn("OMP_NUM_THREADS='8'",text)


if __name__ == '__main__':
    unittest.main()
