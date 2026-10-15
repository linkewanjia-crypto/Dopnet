"""Independent Lee prepare/train orchestration; original epoch mathematics retained."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from .common import (require, digest, file_hash, sign, read_signed, write_json, protect_output,
                     scan_commits, checkpoint_path, derive_summary, publish_commit)
from .preparation import (HERE,SEEDS,VALIDATION_SHA256,RUNTIME_SHA256,RECIPE,runtime_sources,
                          implementation_sources,support_sources,build_config,validate_manifest)
from .scientific_train import (bind_policy,check_deadline,utc_now,seed_lock,load_runtime,seed_runtime,
        environment,branch_tensor_hash,adam,adam_step_counts,rng_state,restore_rng,train_epoch,
        load_validation,validate,binding_for,validate_resume,online_step,chain_rule_scalar_gate)
from .protocol import phase_for_epoch,check_terminal_ledger
from .metadata import verify as verify_metadata
from .policy import RunPolicy,PolicyStopped
DeadlineReached = PolicyStopped

def prepare(args):
    from dopnet_public_reference import check_reference
    output = protect_output(args.output,args.project,args.verification_root,args.reference_results,
                            args.validation_cache,args.configs_dir,HERE)
    policy = RunPolicy(project=args.project,output=output,authorization_text=args.authorization_text,
                       acknowledge_new_run=args.acknowledge_new_run,stop_file=args.stop_file,
                       deadline_utc=args.deadline_utc,run_kind='prepare')
    hardware = policy.resource_check()
    policy.initialize_output(hardware)
    policy.check()
    _,experiment,evaluation = check_reference(args.reference_results)
    sources = runtime_sources(args.project)
    require(experiment['code_sha256']==RUNTIME_SHA256 and evaluation['experiment_signature']==experiment['signature']
            and experiment['validation_cache_sha256']==VALIDATION_SHA256,'Original scientific provenance differs')
    cache = Path(args.validation_cache).resolve()
    require(cache.stat().st_size==4682930065 and file_hash(cache)==VALIDATION_SHA256,'Original fixed validation cache required')
    records={}
    for seed in SEEDS:
        policy.check()
        source = Path(args.configs_dir)/f'lee_seed{seed}.json'
        raw = source.read_bytes()
        cfg = json.loads(raw)
        require(cfg == build_config(experiment['configs'][f'lee3ddual_seed{seed}'],seed,
                                    cfg['lee_meta_reproduction']['microbatch']), 'Packaged science config differs')
        relative = f'configs/lee_meta_seed{seed}.json'
        path = output/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        records[f'lee_meta_seed{seed}'] = dict(file=relative,sha256=file_hash(path),config_sha256=digest(cfg))
    manifest=sign(dict(schema='dopnet_lee_research_manifest_v1',recipe=RECIPE,seeds=list(SEEDS),
        new_run_id=policy.run_id,created_utc=utc_now().isoformat(),historical_execution_authorization=False,
        reference_experiment_signature=experiment['signature'],reference_evaluation_signature=evaluation['signature'],
        runtime_sources_sha256=sources,implementation_sha256=implementation_sources(),
        verification_root=str(Path(args.verification_root).resolve()),
        verification_support_sha256=support_sources(args.verification_root),validation_cache_sha256=VALIDATION_SHA256,
        validation=dict(path=str(cache),bytes=4682930065,frames=2000,optimizer_use=False),configs=records))
    policy.check();write_json(output/'manifest.json',manifest)
    return 0

def qualify(manifest,cfg,project,output):
    args=SimpleNamespace(project=project,microbatch=cfg['lee_meta_reproduction']['microbatch'])
    result=dict(schema='dopnet_lee_research_actual_cuda_probe_v1',status='failed',
                manifest_signature=manifest['signature'],implementation_sha256=manifest['implementation_sha256'],
                recipe_sha256=cfg['lee_meta_reproduction']['recipe_sha256'],microbatch=args.microbatch,
                synthetic_hardware_only=True,scientific_data_generated=False,validation_cache_opened=False,
                fullpath_qualification=False,exact_fullpath_peak_verified=False,training_started=False,
                historical_parent_receipt_authorization=False,observed_start_utc=utc_now().isoformat())
    try:
        torch, _, _, module, updates = load_runtime(args.project, hardware_only=True)
        result['environment'] = environment(torch)
        result['scalar_full_chain_rule_gradient'] = chain_rule_scalar_gate(torch)
        generator = torch.Generator(device='cuda:0').manual_seed(970000043)
        inputs = torch.randn((32, 2, 5, 25, 8, 128), generator=generator, device='cuda:0')
        truth = torch.randn((32, 2, 2, 25, 8, 128), generator=generator, device='cuda:0')
        model = module.LeeAlternativeMeta().to('cuda:0')
        sft_before = branch_tensor_hash(model)
        torch.cuda.reset_peak_memory_stats()
        records = {}
        for phase in ('add_meta', 'joint_online'):
            optimizer = adam(torch, model) if phase == 'joint_online' else None
            torch.cuda.synchronize()
            started = time.perf_counter()
            losses = []
            for _ in range(2):
                check_deadline()
                if phase == 'add_meta':
                    metrics = updates.add_meta_step(model, inputs[:16], truth[:16], inputs[16:], truth[16:],
                                                   microbatch=args.microbatch, deadline_check=check_deadline)
                    losses.append(metrics['query_loss'])
                else:
                    losses.append(online_step(torch, module, model, optimizer, inputs, truth, args.microbatch))
            torch.cuda.synchronize()
            records[phase] = {'actual_updates': 2, 'seconds': time.perf_counter()-started,
                              'finite_losses': losses}
            if phase == 'add_meta' and branch_tensor_hash(model) != sft_before:
                raise ValueError('Probe offline stage changed SFT weights')
        check_deadline()
        peak = int(torch.cuda.max_memory_reserved())
        if peak > 6*2**30:
            raise ValueError('Probe exceeded6GiB Torch allocation budget')
        result.update(status='passed', phases=records,
                      offline_sft_unchanged=True, torch_peak_reserved_bytes=peak,
                      torch_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                      expected_exclusive_seconds_per12000_frames={p: r['seconds']*375/2 for p,r in records.items()})
    except Exception:
        result['failure_traceback'] = traceback.format_exc()
        raise
    finally:
        result['observed_end_utc'] = utc_now().isoformat()
        write_json(output, sign(result))

    return result

def train(args):
    manifest,cfg,name = validate_manifest(args.manifest,args.project,args.seed,args.verification_root)
    output = protect_output(args.output,args.project,args.verification_root,args.manifest,HERE,
                            manifest['validation']['path'])
    policy = RunPolicy(project=args.project,output=output,authorization_text=args.authorization_text,
                       acknowledge_new_run=args.acknowledge_new_run,resume=args.resume,
                       stop_file=args.stop_file,deadline_utc=args.deadline_utc,run_kind='train',
                       min_free_gpu_memory_mib=38400)
    hardware=policy.resource_check();policy.initialize_output(hardware)
    bind_policy(policy);check_deadline()
    sys.path.insert(0,str(Path(args.verification_root).resolve()))
    folder=output/'checkpoints'/name
    binding=dict(binding_for(manifest,cfg,args.seed),new_run_id=policy.run_id)
    with seed_lock(folder):
        records = scan_commits(folder, binding)
        if records and not args.resume:
            raise ValueError('Existing full commits require explicit --resume')
        if args.resume:
            # Keep prior diagnostics intact while making this explicit new
            # attempt's terminal status unambiguous. All destinations stay here.
            import uuid
            for name in ('failure.json', 'incomplete.json'):
                path = folder / name
                if path.exists():
                    old = folder / 'prior_attempts' / (uuid.uuid4().hex + '_' + name)
                    old.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(old)
        if len(records) == 40:
            proof = verify_metadata(folder, binding, cfg, args.checkpoint_reader)
            write_json(folder / 'TRAINING_METADATA_VERIFIED.json', proof)
            return 0
        import uuid
        qualification = qualify(manifest,cfg,args.project,output/f'CUDA_QUALIFICATION_{args.seed}_{uuid.uuid4().hex}.json')
        require(qualification['status']=='passed','Actual fresh CUDA qualification failed')
        # The synthetic probe is not part of the measured waveform path.
        # Reset its allocator cache; training then reseeds all original RNGs.
        sys.modules['torch'].cuda.empty_cache()
        torch, trainer, receiver, module, updates = load_runtime(args.project)
        seed_runtime(torch, args.seed)
        model = module.LeeAlternativeMeta().to('cuda:0')
        env = environment(torch)
        initial_sft = branch_tensor_hash(model)
        optimizer = None
        summary = derive_summary(records)
        history = summary['history']
        state = None
        if records:
            state = torch.load(checkpoint_path(folder, len(records)), map_location='cuda:0', weights_only=False)
            validate_resume(state, records[-1], history, binding, cfg, env)
            model.load_state_dict(state['model'], strict=True)
            initial_sft = state['initial_sft_sha256']
            if len(records) >= 2:
                optimizer = adam(torch, model)
                optimizer.load_state_dict(state['optimizer'])
        generator = trainer.BatchChannelGenerator(cfg, device='cuda:0', seed=args.seed + 10001)
        validation = load_validation(cfg, manifest, folder)
        if state is not None:
            restore_rng(torch, state)
        del state
        try:
            for epoch in range(len(records) + 1, 41):
                check_deadline()
                if epoch == 2:
                    if optimizer is not None:
                        raise ValueError('Joint Adam must initialize exactly once at global2')
                    optimizer = adam(torch, model)
                    print('LEE_META_TO_JOINT global_epoch=2 data_epoch=2 preserve_global_RNG=true', flush=True)
                torch.cuda.reset_peak_memory_stats()
                row = train_epoch(torch, module, updates, model, optimizer, generator, cfg, epoch)
                row.update(validate(torch, receiver, model, validation, cfg, epoch))
                row['torch_peak_allocated_bytes'] = max(row['fullpath_torch_allocated_peak_bytes'],int(torch.cuda.max_memory_allocated()))
                row['torch_peak_reserved_bytes'] = max(row['fullpath_torch_reserved_peak_bytes'],int(torch.cuda.max_memory_reserved()))
                if epoch == 1 and branch_tensor_hash(model) != initial_sft:
                    raise ValueError('SFT changed during ADD-only offline training')
                history = history + [row]
                # Recheck frozen runtime/recipe at the full-epoch commit boundary.
                again, cfg_again, _ = validate_manifest(args.manifest, args.project, args.seed, args.verification_root)
                if again != manifest or cfg_again != cfg:
                    raise ValueError('Provenance changed during an epoch')
                committed = {'schema': 'dopnet_lee_research_full_state_v1', 'binding': binding, 'cfg': cfg,
                             'epoch': epoch, 'data_epoch': epoch, 'phase': phase_for_epoch(epoch),
                             'history': history, 'model': model.state_dict(),
                             'optimizer': optimizer.state_dict() if optimizer is not None else None,
                             'optimizer_name': 'explicit_GD' if epoch == 1 else 'Adam',
                             'adam_step_counts': adam_step_counts(optimizer) if optimizer is not None else [],
                             'meta_parent_sha256': records[0]['state_sha256'] if records else None,
                             'initial_sft_sha256': initial_sft, 'environment': env,
                             'generator_identity': {'seed': args.seed+10001, 'last_global_epoch': epoch,
                                                    'next_global_epoch': epoch+1, 'generation_batch': 32,
                                                    'link_batch': 8, 'stateless_per_epoch': True},
                             'state_created_utc': utc_now().isoformat(), **rng_state(torch)}
                record = publish_commit(folder, committed, binding, torch.save, check_deadline)
                records.append(record)
                summary = derive_summary(records)
                write_json(folder / 'progress.json', sign(summary))
                print(f'LEE_COMMIT seed={args.seed} epoch={epoch}/40 val_mse={row["val_mse"]:.9g} sha={record["state_sha256"]}', flush=True)
                del committed
            check_deadline()
            check_terminal_ledger(history)
            write_json(folder / 'TRAINING_SUBMITTED40.json', sign(dict(summary,
                       status='submitted40_pending_actual_metadata_verifier',
                       training_delivery_qualified=False, all_experiments_complete=False,
                       finished_utc=utc_now().isoformat(), binding=binding)))
            proof = verify_metadata(folder, binding, cfg, args.checkpoint_reader)
            write_json(folder / 'TRAINING_METADATA_VERIFIED.json', proof)
            return 0
        except DeadlineReached:
            write_json(folder / 'incomplete.json', sign(dict(derive_summary(records),
                       status='incomplete', exit_code=75, reason='new_run_stop_or_deadline',
                       current_partial_epoch_discarded=True, observed_utc=utc_now().isoformat())))
            return 75
        except Exception:
            write_json(folder / 'failure.json', sign(dict(derive_summary(records),
                       status='incomplete', exit_code=1, traceback=traceback.format_exc(),
                       observed_utc=utc_now().isoformat())))
            raise
