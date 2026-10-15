"""CPU-only gates for archived-checkpoint and matched-stream reuse."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from experiment_protocol import SEEDS, make_config
from historical_reuse import (HistoricalTrainingReference, OLD_VARIANTS,
                              NEW_CONFIG_FIELDS, mixed_configs,
                              project_archived_checkpoint_config)


class HistoricalReuseTest(unittest.TestCase):
    def test_only_three_metadata_fields_may_be_added_to_old_configs(self):
        base=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/dopnet.yaml')
                            .read_text(encoding='utf-8'))
        old={}
        for variant in OLD_VARIANTS:
            for seed in SEEDS:
                cfg=make_config(base,variant,seed)
                for section,field in NEW_CONFIG_FIELDS:
                    cfg[section].pop(field,None)
                old[f'{variant}_seed{seed}']=cfg
        reference=SimpleNamespace(experiment={'configs':old})
        self.assertEqual(len(mixed_configs(base,reference)),48)
        altered=copy.deepcopy(old)
        altered['dopnet_seed42']['train']['lr']*=2
        with self.assertRaisesRegex(RuntimeError,'Old/new training configuration'):
            mixed_configs(base,SimpleNamespace(experiment={'configs':altered}))

    def test_all_original_batch_hashes_are_required(self):
        old_identity={'seed':42,'epoch':1,'code_sha256':'a'*64,
                      'system':{'frame_mode':'dual_domain_v4'},
                      'data_pipeline':{'emit_pilot_ls':True}}
        current_identity=copy.deepcopy(old_identity)
        current_identity['code_sha256']='b'*64
        current_identity['system']['frame_mode']='dopnet'
        current_identity['data_pipeline']['emit_pilot_only_fast_time_add']=True
        old={'identity':old_identity,'batches':[{'tensor_sha256':'c'*64} for _ in range(375)],
             'data_signature':'d'*64}
        current={'identity':current_identity,'batches':[
            {'legacy_tensor_sha256':'c'*64} for _ in range(375)],
            'data_signature':'e'*64}
        historical=HistoricalTrainingReference.__new__(HistoricalTrainingReference)
        historical.old_code_sha256='a'*64
        historical.manifests={(42,1):(old,'f'*64)}
        proof=historical.check_epoch(current,42,1)
        self.assertEqual(proof['matched_batches'],375)
        current['batches'][374]['legacy_tensor_sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'batch 374'):
            historical.check_epoch(current,42,1)

    def test_archived_name_projection_rejects_distilled_lite(self):
        base=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/dopnet.yaml')
                            .read_text(encoding='utf-8'))
        current=make_config(base,'dopnet_lite',42)
        for section,field in NEW_CONFIG_FIELDS:
            current[section].pop(field,None)
        raw=copy.deepcopy(current)
        raw['system']['frame_mode']='dual_domain_v4'
        raw['model']['estimator']='v4_dual3d'
        raw['model']['ablation_profile']='student_v1'
        raw['distill']={'enabled':False,'cfr_weight':0.5}
        self.assertEqual(project_archived_checkpoint_config(raw,'dopnet_lite'),current)
        raw['distill']['enabled']=True
        with self.assertRaisesRegex(RuntimeError,'no-distillation'):
            project_archived_checkpoint_config(raw,'dopnet_lite')


if __name__=='__main__':
    unittest.main()
