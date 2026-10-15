"""New-run Lee inference with original frozen score loop and receiver scope."""
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile
from .common import (require, digest, file_hash, read_signed, sign, write_json, protect_output,
                     scan_commits, checkpoint_path, derive_summary)
from .preparation import validate_manifest, build_config
from .scientific_train import bind_policy, check_deadline, load_runtime, binding_for
from .score_support import CLAIM, NEW_METHOD, SEEDS, WIMAE_ARCHIVE_SHA, require_routes, read_parent_archive
from . import score_support as Recovery
from .metadata import verify as verify_metadata
from .policy import RunPolicy
signed = read_signed
write = write_json

def evaluate(args):
    from dopnet_public_reference import ReferenceResults, check_reference
    out = protect_output(args.output,args.project,args.verification_root,args.reference_results,
                         args.training_output,args.manifest,args.parent_archive,Path(__file__).parent)
    policy = RunPolicy(project=args.project, output=out, authorization_text=args.authorization_text,
                       acknowledge_new_run=args.acknowledge_new_run, stop_file=args.stop_file,
                       deadline_utc=args.deadline_utc, run_kind='evaluate', min_free_gpu_memory_mib=38400)
    hardware = policy.resource_check()
    policy.initialize_output(hardware)
    bind_policy(policy)
    check_deadline()
    parent_ev,parent_delivery = read_parent_archive(args.parent_archive)
    reference_root,experiment,old_ev = check_reference(args.reference_results)
    trained = {}
    for seed in SEEDS:
        m,cfg,name = validate_manifest(args.manifest,args.project,seed,args.verification_root)
        require(cfg == build_config(experiment['configs']['lee3ddual_seed'+str(seed)],seed,
                                   cfg['lee_meta_reproduction']['microbatch']), 'Reviewed science config differs')
        folder = Path(args.training_output)/'checkpoints'/name
        authorization = signed(Path(args.training_output)/'RUN_AUTHORIZATION.json')
        binding = dict(binding_for(m,cfg,seed),new_run_id=authorization['run_id'])
        proof = verify_metadata(folder,binding,cfg,args.checkpoint_reader)
        stored = signed(folder/'TRAINING_METADATA_VERIFIED.json')
        require(proof == stored, 'Actual newly verified metadata differs from submitted receipt')
        records = scan_commits(folder,binding)
        selected = records[proof['selected_epoch']-1]
        trained[seed] = dict(cfg=cfg,folder=folder,proof=proof,selected=selected)
    require(m['reference_experiment_signature'] == experiment['signature'] == parent_ev['reference_experiment_signature']
            and m['reference_evaluation_signature'] == old_ev['signature'] == parent_ev['reference_evaluation_signature'],
            'Train/accepted parent/frozen reference identity differs')
    routes = parent_ev['routes']+[dict(method=NEW_METHOD,training_seed=s,trained=True) for s in SEEDS]
    require_routes(routes,parent_ev['routes'])
    sys.path.insert(0,str(Path(args.verification_root).resolve()))
    torch,_,receiver,model_module,_ = load_runtime(args.project)
    import numpy as np
    from verification_extension import protocol as P
    from reference_replay import update_waveform_digest
    from data.generator import BatchChannelGenerator
    from utils.metrics import nmse_per_sample
    from evaluate import frame_bit_counts
    from experiment_protocol import TEST_SEED, experiment_cells
    from aggregate_results import aggregate
    reference_root,exp_again,old_again = check_reference(args.reference_results)
    require(exp_again == experiment and old_again == old_ev,'Reference changed')
    reference = ReferenceResults(reference_root)
    cfg = copy.deepcopy(experiment['configs']['dopnet_seed42']);cfg['model']['ablation_profile']='full'
    all_cells,skipped = experiment_cells(cfg)
    reference.validate_protocol(experiment,cfg,all_cells,skipped)
    cells = P.select_suite(all_cells,'manuscript')
    require(cells == parent_ev['cells'] and len(cells)==269,'Do not generate another test suite')
    models=[]
    for seed in SEEDS:
        check_deadline()
        state = torch.load(checkpoint_path(trained[seed]['folder'],trained[seed]['proof']['selected_epoch']),
                           map_location='cpu',weights_only=False)
        require(state['cfg']==trained[seed]['cfg'] and state['epoch']==trained[seed]['proof']['selected_epoch'],
                'Selected model metadata differs')
        model = model_module.LeeAlternativeMeta().to('cuda:0').eval()
        model.load_state_dict(state['model'],strict=True);models.append(model);del state
    payload = dict(schema='dopnet_lee_research_frozen_evaluation_v1',routes=routes,cells=cells,suite='manuscript',
        samples_per_cell=512,batch_size=8,inference_microbatch=args.inference_microbatch,inference_precision='fp32',
        test_seed=TEST_SEED,reference_experiment_signature=experiment['signature'],
        reference_evaluation_signature=old_ev['signature'],fixed_test_reference=reference.provenance(),
        parent_archive_sha256=file_hash(args.parent_archive),source_parent_archive_sha256=WIMAE_ARCHIVE_SHA,
        parent_data_manifest_signature=parent_delivery['signature'],
        historical_parent_evaluation_signature=parent_ev['signature'],inherited_routes=21,
        training_manifest_sha256=file_hash(args.manifest),training_manifest_signature=m['signature'],
        training_receipts={str(s):trained[s]['proof'] for s in SEEDS},
        checkpoint_sha256={str(s):trained[s]['selected']['state_sha256'] for s in SEEDS},
        implementation_sha256=m['implementation_sha256'],claim=CLAIM,original_author_dataset_recipe_reproduction=False,
        original_algorithm='Algorithm1_full_second_order_ADD_only_then_parallel_SFT_plus_ADD_joint',
        task_adaptation=dict(meta_epochs=1,online_epochs=39,fresh_frames_per_epoch=12000,updates_per_epoch=375,
                             alpha=.001,beta=.001,K=1,support=16,query=16),
        test_adaptation_steps=0,receiver_protocols=['common_diagonal','native_own_ici_else_diagonal'],
        new_run_id=policy.run_id,historical_delivery=False)
    ev = sign(payload)
    with P.writer_lock(out,'.lee_evaluation.lock'), ZipFile(args.parent_archive) as z:
        write(out/'evaluation.json',ev)
        generator = BatchChannelGenerator(cfg,torch.device('cuda:0'),TEST_SEED)
        for number,cell in enumerate(cells,1):
            check_deadline(); cid=cell['cell_id']; path=out/'cells'/f'{cid}.npz'; marker=path.with_suffix('.COMPLETE.json')
            parent_name=f'evaluation/cells/{cid}.npz'
            data=z.read(parent_name); parent_sha=hashlib.sha256(data).hexdigest()
            require(parent_sha == parent_delivery['files'][parent_name],'Parent cell changed')
            old_sha=reference.source_score_sha256(cell)
            with np.load(io.BytesIO(data),allow_pickle=False) as old:
                require(old['signature'].item() == parent_ev['signature'] and json.loads(old['meta'].item()) == cell and old['source_score_sha256'].item() == old_sha,'Inherited cell reference changed')
                expected_wave=old['realization_sha256'].item();reference.assert_fingerprint(cell,expected_wave)
                require(np.array_equal(old['cluster_id'],np.arange(512)//8),'Parent cluster identity differs')
                arrays={'nmse':np.empty((24,512),dtype=np.float64),'native_available':np.zeros(24,dtype=np.int8)}
                arrays['nmse'][:21]=old['nmse']; arrays['native_available'][:21]=old['native_available']
                expected_bits=None
                if cell['ber']:
                    for key in Recovery.receiver_keys(cell,old.files):
                        arrays[key]=np.zeros((24,512),dtype=old[key].dtype);arrays[key][:21]=old[key]
                    expected_bits=old['bits'][0].copy();require(np.all(old['bits']==expected_bits),'Inherited bit denominators differ')
            if path.exists() and marker.exists():
                done=signed(marker)
                require(done['evaluation_signature']==ev['signature'] and done['score_sha256']==file_hash(path) and done['parent_score_sha256']==parent_sha and done['source_score_sha256']==old_sha and done['realization_sha256']==expected_wave,'Invalid saved cell')
                continue
            waveform=hashlib.sha256()
            with torch.inference_mode():
                for start in range(0,512,8):
                    check_deadline(); draw_seed=TEST_SEED+int(digest(dict(cell=cid,offset=start))[:15],16)
                    rng=torch.Generator(device='cuda:0').manual_seed(draw_seed)
                    cond=dict(channel_type=cell['channel'],modulation=cell['modulation'],delay_spread_ns=cell['delay_spread_ns'],doppler_hz=cell['doppler_hz'],cp_len=cell['cp_len'],pilot_spacing=cell['pilot_spacing'])
                    batch=generator.simulate(8,cond,torch.full((8,),cell['snr_db'],device='cuda:0'),rng,draw_seed+1000000001)
                    update_waveform_digest(waveform,batch)
                    for offset in range(0,8,args.inference_microbatch):
                        stop=min(offset+args.inference_microbatch,8); frames=slice(start+offset,start+stop)
                        target=receiver.select_data_csi(batch['target_h'][offset:stop],cfg)
                        for index,(seed,model) in enumerate(zip(SEEDS,models),21):
                            check_deadline(); prediction=receiver.select_data_csi(model(batch['input'][offset:stop]),trained[seed]['cfg'])
                            nmse=nmse_per_sample(prediction,target).cpu().numpy()
                            require(nmse.shape==(stop-offset,) and np.isfinite(nmse).all() and np.all(nmse>=0),'Invalid Lee prediction NMSE')
                            arrays['nmse'][index,frames]=nmse
                            if cell['ber']:
                                symbols,_=receiver.linear_multiuser_equalize(prediction,batch['rx_data'][offset:stop],batch['noise_var'][offset:stop],method='mmse')
                                require(bool(torch.isfinite(symbols).all()),'Nonfinite Lee receiver')
                                errors,bits=frame_bit_counts(symbols,batch['tx_label'][offset:stop],cell['modulation'])
                                require(np.array_equal(bits,expected_bits[frames]),'Frozen actual bit counts differ')
                                arrays['bits'][index,frames]=bits
                                arrays['errors_diagonal'][index,frames]=arrays['errors_native'][index,frames]=errors
                                for db,key in ((-3,'errors_noise_minus3db'),(3,'errors_noise_plus3db')):
                                    if key not in arrays:continue
                                    changed,_=receiver.linear_multiuser_equalize(prediction,batch['rx_data'][offset:stop],batch['noise_var'][offset:stop]*10**(db/10),method='mmse')
                                    require(bool(torch.isfinite(changed).all()),'Nonfinite noise sensitivity receiver')
                                    arrays[key][index,frames]=frame_bit_counts(changed,batch['tx_label'][offset:stop],cell['modulation'])[0]
                            del prediction
                        del target
                    del batch
            require(waveform.hexdigest()==expected_wave,'Full frozen waveform SHA differs')
            reference.assert_fingerprint(cell,waveform.hexdigest());check_deadline()
            arrays.update(cluster_id=np.arange(512)//8,signature=np.array(ev['signature']),meta=np.array(json.dumps(cell)),realization_sha256=np.array(expected_wave),source_score_sha256=np.array(old_sha),parent_score_sha256=np.array(parent_sha))
            P.atomic_npz(path,arrays)
            write(marker,sign(dict(evaluation_signature=ev['signature'],score_sha256=file_hash(path),source_score_sha256=old_sha,parent_score_sha256=parent_sha,realization_sha256=expected_wave)))
            print(f'[{number}/269] Lee3 frozen SHA PASS {cid}',flush=True)

        write(out/'evaluation_complete.json',dict(signature=ev['signature'],cells=269))
        check_deadline();aggregate(out,2000);check_deadline()
        write(out/'COMPLETE.json',sign(dict(schema='dopnet_lee_research_scoring_complete_v1',status='complete',
              evaluation_signature=ev['signature'],cells=269,paper_ready=False,new_run_id=policy.run_id,
              historical_delivery=False,finished_utc=datetime.now(timezone.utc).isoformat())))
    return 0
