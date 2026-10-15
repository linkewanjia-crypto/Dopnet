"""Server-only Lee3 inference, exact reuse of accepted parent21 frozen scores."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from zipfile import ZipFile
from .protocol import (CLAIM, DEADLINE, NEW_METHOD, SEEDS, DeadlineReached,
    bind_training_source, check_deadline, digest, file_hash, no_live_training,
    read, read_parent_archive, require, require_routes, require_training_receipt,
    sign, signed, verify_source, write, WIMAE_ARCHIVE_SHA)
from .execution import bind_args, cli_arguments
from . import recovery as Recovery


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'reference-results', 'training-output', 'training-code-root', 'training-queue', 'parent-archive', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--recovery-parent',type=Path,required=True)
    ap.add_argument('--inference-microbatch', type=int, choices=(1,2,4,8), default=1)
    cli_arguments(ap)
    args = ap.parse_args()
    require(os.name == 'posix', 'Real inference only on authorized CUDA server')
    execution_binding, queue = bind_args(args)
    recovery_binding,recovery_files,recovery_evaluation=Recovery.inspect_stopped(args.recovery_parent)
    check_deadline(); no_live_training()
    import subprocess
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    require(not gpu.stdout.strip(),'Keep all real GPU work serial; another compute process is active')
    own = verify_source(Path(__file__).resolve().parent.parent)
    training_source = bind_training_source(args.training_code_root)
    parent_ev, parent_delivery = read_parent_archive(args.parent_archive)
    require(Path(queue['selection']['manifest']).resolve() == (args.training_output/'lee_manifest.json').resolve(), 'Wrong selected training output')
    sys.path.insert(0, str(args.training_code_root.resolve()))
    from lee_meta_equal40.prepare import validate_manifest, build_config
    from lee_meta_equal40.artifacts import scan_commits, checkpoint_path
    from lee_meta_equal40.train import binding_for
    experiment = signed(args.reference_results/'experiment.json')
    old_ev = signed(args.reference_results/'evaluation.json')
    trained = {}
    for seed in SEEDS:
        m, cfg, name = validate_manifest(args.training_output/'lee_manifest.json', args.project, seed)
        require(cfg == build_config(experiment['configs']['lee3ddual_seed'+str(seed)], seed, queue['selection']['microbatch']), 'Training config differs from reviewed adaptation')
        folder = args.training_output/'checkpoints'/name
        proof_path = folder/'TRAINING_METADATA_VERIFIED.json'
        proof = signed(proof_path)
        require(file_hash(proof_path) == queue['tasks']['verify_seed'+str(seed)]['proof_sha256'], 'Actual metadata receipt changed')
        require_training_receipt(proof, seed, m, cfg, execution_binding)
        records = scan_commits(folder, binding_for(m,cfg,seed))
        require(len(records) == 40 and not (folder/'failure.json').exists() and not (folder/'incomplete.json').exists(), 'Incomplete immutable commit ledger')
        selected = records[proof['selected_epoch']-1]
        require(selected['state_sha256'] == proof['selected_state_sha256'] and records[-1]['state_sha256'] == proof['terminal_state_sha256'], 'Selected/terminal bytes differ')
        trained[seed] = dict(cfg=cfg, folder=folder, proof=proof, selected=selected)
    require(m['reference_experiment_signature'] == experiment['signature'] == parent_ev['reference_experiment_signature']
            and m['reference_evaluation_signature'] == old_ev['signature'] == parent_ev['reference_evaluation_signature'], 'Train/parent/frozen reference identity differs')
    routes = parent_ev['routes']+[dict(method=NEW_METHOD,training_seed=s,trained=True) for s in SEEDS]
    require_routes(routes,parent_ev['routes'])
    check_deadline(); no_live_training()
    from lee_meta_equal40.train import load_runtime
    torch, _, receiver, model_module, _ = load_runtime(args.project)
    sys.path.insert(0,str(args.project.resolve()))
    import numpy as np
    from verification_extension import protocol as P
    from reference_replay import ReferenceResults, update_waveform_digest
    from data.generator import BatchChannelGenerator
    from utils.metrics import nmse_per_sample
    from evaluate import frame_bit_counts
    from experiment_protocol import TEST_SEED, experiment_cells
    from aggregate_results import aggregate
    reference_root, exp_again, old_again = P.check_reference(args.reference_results)
    require(exp_again == experiment and old_again == old_ev, 'Reference changed')
    reference = ReferenceResults(reference_root)
    cfg = copy.deepcopy(experiment['configs']['dopnet_seed42']); cfg['model']['ablation_profile']='full'
    all_cells, skipped = experiment_cells(cfg)
    reference.validate_protocol(experiment,cfg,all_cells,skipped)
    cells = P.select_suite(all_cells,'manuscript')
    require(cells == parent_ev['cells'] and len(cells) == 269, 'Do not generate another test suite')
    models = []
    for seed in SEEDS:
        check_deadline()
        state = torch.load(checkpoint_path(trained[seed]['folder'],trained[seed]['proof']['selected_epoch']),map_location='cpu',weights_only=False)
        require(state['cfg'] == trained[seed]['cfg'] and state['epoch'] == trained[seed]['proof']['selected_epoch'], 'Actual selected model metadata differs')
        model = model_module.LeeAlternativeMeta().to('cuda:0').eval()
        model.load_state_dict(state['model'],strict=True); models.append(model); del state
    payload = dict(schema='lee_equal40_frozen_evaluation_v1',routes=routes,cells=cells,suite='manuscript',
        samples_per_cell=512,batch_size=8,inference_microbatch=args.inference_microbatch,inference_precision='fp32',test_seed=TEST_SEED,
        reference_experiment_signature=experiment['signature'],reference_evaluation_signature=old_ev['signature'],fixed_test_reference=reference.provenance(),
        parent_archive_sha256=WIMAE_ARCHIVE_SHA,parent_evaluation_signature=parent_ev['signature'],inherited_routes=21,
        training_manifest_sha256=file_hash(args.training_output/'lee_manifest.json'),training_manifest_signature=m['signature'],
        training_source_signature=training_source['signature'],training_queue_sha256=file_hash(args.training_queue),
        training_receipts={str(s):trained[s]['proof'] for s in SEEDS},
        checkpoint_sha256={str(s):trained[s]['selected']['state_sha256'] for s in SEEDS},
        implementation_signature=own['signature'],claim=CLAIM,original_author_dataset_recipe_reproduction=False,
        original_algorithm='Algorithm1_full_second_order_ADD_only_then_parallel_SFT_plus_ADD_joint',
        task_adaptation=dict(meta_epochs=1,online_epochs=39,fresh_frames_per_epoch=12000,updates_per_epoch=375,alpha=.001,beta=.001,K=1,support=16,query=16),
        test_adaptation_steps=0,receiver_protocols=['common_diagonal','native_own_ici_else_diagonal'],
        scoring_recovery=recovery_binding,**execution_binding)
    ev = sign(payload); out = args.output.resolve()
    for protected in (args.project,args.reference_results,args.training_output,args.recovery_parent,Path(__file__).resolve().parent.parent,Recovery.OLD_CODE,Recovery.CONTROL,Recovery.LAUNCH):
        protected=protected.resolve();require(out != protected and protected not in out.parents,'Use independent scoring output')
    require(not out.exists() or not any(out.iterdir()),'Unique recovery requires fresh independent output; retain any failed candidate')
    out.mkdir(parents=True,exist_ok=True)
    with P.writer_lock(out,'.lee_evaluation.lock'), ZipFile(args.parent_archive) as z:
        if (out/'evaluation.json').exists():
            require(signed(out/'evaluation.json') == ev,'Do not resume another scoring identity')
        write(out/'evaluation.json',ev)
        queue_bytes=args.training_queue.read_bytes()
        require(hashlib.sha256(queue_bytes).hexdigest()==ev['training_queue_sha256'],'Training queue changed after its natural completion')
        (out/'training_queue_state.json').write_bytes(queue_bytes)
        Recovery.reuse_completed(out,recovery_binding,recovery_files,recovery_evaluation,ev)
        generator = BatchChannelGenerator(cfg,torch.device('cuda:0'),TEST_SEED)
        for number,cell in enumerate(cells,1):
            check_deadline(); cid=cell['cell_id']; path=out/'cells'/f'{cid}.npz'; marker=path.with_suffix('.COMPLETE.json')
            parent_name=f'evaluation/cells/{cid}.npz'
            data=z.read(parent_name); parent_sha=hashlib.sha256(data).hexdigest()
            require(parent_sha == parent_delivery['files'][parent_name],'Parent cell changed')
            old_sha=file_hash(reference_root/'cells'/f'{cid}.npz')
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
        write(out/'COMPLETE.json',sign(dict(status='complete',evaluation_signature=ev['signature'],cells=269,paper_ready=False,pending=['paper_assets'],finished_utc=datetime.now(timezone.utc).isoformat(),deadline_utc=None,legacy_deadline_utc=DEADLINE,**execution_binding)))


if __name__ == '__main__':
    try:
        main()
    except DeadlineReached as error:
        print('LEE_SCORING_INCOMPLETE_EXPLICIT_STOP '+str(error),file=sys.stderr,flush=True)
        raise SystemExit(75)
