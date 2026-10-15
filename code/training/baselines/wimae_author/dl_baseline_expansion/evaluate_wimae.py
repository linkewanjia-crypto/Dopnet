"""Three-seed WiMAE scoring on the unchanged269 frozen512/8 test cells.

The15 historical core columns and3 completed equal-budget FortiTran columns
are copied with exact source hashes; only the3 new WiMAE predictions are run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

from .prepare_wimae import digest, file_hash, validate_manifest, write_json
from .train_wimae import DeadlineReached, check_deadline, deadline_from
from .wimae_protocol import (CLAIM, FORTI_METHOD, NEW_METHOD, SEEDS, check_training,
                             core_routes, read_json, require)


def check_forti_reference(folder, reference_root, experiment, old_evaluation, cells, implementation_root=None):
    from budget_matched_baselines import verify_author_delivery as delivery
    # Full metadata/CRC/SHA delivery verification, including authenticated40
    # epoch training and the original score columns. No model execution here.
    archive = Path(folder) / "budget_matched_results.zip"
    implementation_root = Path(implementation_root or delivery.__file__).resolve()
    if implementation_root.is_file():
        implementation_root = implementation_root.parent
    proof = delivery.verify(archive, reference_root, implementation_root)
    with ZipFile(archive) as z:
        archive_hashes = json.loads(z.read("DELIVERY_MANIFEST.json"))["files"]
    require(file_hash(Path(folder) / "evaluation.json") == archive_hashes["evaluation/evaluation.json"],
            "External Forti evaluation metadata differs from the verified delivery ZIP")
    ev = read_json(Path(folder) / "evaluation.json")
    require(ev.get("reference_experiment_signature") == experiment["signature"]
            and ev.get("reference_evaluation_signature") == old_evaluation["signature"]
            and ev.get("cells") == cells and ev.get("samples_per_cell") == 512
            and ev.get("batch_size") == 8, "FortiTran comparison uses another frozen test suite")
    indices = [i for i, r in enumerate(ev["routes"]) if r["method"] == FORTI_METHOD]
    require([ev["routes"][i]["training_seed"] for i in indices] == list(SEEDS),
            "Completed fixed-budget FortiTran lacks3seeds")
    for cell in cells:
        for suffix in (".npz", ".COMPLETE.json"):
            relative = "cells/" + cell["cell_id"] + suffix
            require(file_hash(Path(folder) / relative) == archive_hashes["evaluation/" + relative],
                    "External Forti score/marker differs from the verified delivery ZIP: " + relative)
    return ev, indices, proof, file_hash(archive), archive_hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "reference-results", "training-output", "forti-evaluation", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--deadline-utc", required=True)
    ap.add_argument("--inference-microbatch", type=int, choices=(1, 2, 4, 8), default=1)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--forti-code-root", type=Path)
    args = ap.parse_args()
    deadline = deadline_from(args.deadline_utc)
    check_deadline(deadline)
    sys.path.insert(0, str(args.project.resolve()))
    import numpy as np
    import torch
    from verification_extension import protocol as P
    from reference_replay import ReferenceResults, update_waveform_digest
    from data.generator import BatchChannelGenerator
    from models.receiver import select_data_csi, linear_multiuser_equalize
    from utils.metrics import nmse_per_sample
    from utils.checkpoints import load_checkpoint
    from evaluate import frame_bit_counts
    from experiment_protocol import TEST_SEED, experiment_cells
    from aggregate_results import aggregate
    from .wimae_author import WiMAESliceCE

    require(torch.cuda.is_available(), "Run on the CUDA server, never the local laptop")
    reference_root, experiment, old_ev = P.check_reference(args.reference_results)
    reference = ReferenceResults(reference_root)
    recipe_path = args.training_output / "wimae_manifest.json"
    recipe, _, _ = validate_manifest(recipe_path, args.project, 42)
    require(recipe["reference_experiment_signature"] == experiment["signature"]
            and recipe["reference_evaluation_signature"] == old_ev["signature"], "WiMAE training reference differs")
    cfg = copy.deepcopy(experiment["configs"]["dopnet_seed42"])
    cfg["model"]["ablation_profile"] = "full"
    all_cells, skipped = experiment_cells(cfg)
    reference.validate_protocol(experiment, cfg, all_cells, skipped)
    cells = P.select_suite(all_cells, "manuscript")
    require(len(cells) == 269, "Current manuscript suite must contain269cells")
    old_indices, old_routes = core_routes(old_ev["routes"])
    forti_ev, forti_indices, forti_proof, forti_zip_sha, forti_archive_hashes = check_forti_reference(
        args.forti_evaluation, reference_root, experiment, old_ev, cells,
        args.forti_code_root)
    forti_routes = [forti_ev["routes"][i] for i in forti_indices]
    routes = old_routes + forti_routes + [dict(method=NEW_METHOD, training_seed=s, trained=True) for s in SEEDS]
    require(len(routes) == 21, "Seven methods must produce21routes")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    models, configs, checkpoint_hashes, training_artifacts = [], [], {}, {}
    for seed in SEEDS:
        check_deadline(deadline)
        _, model_cfg, name = validate_manifest(recipe_path, args.project, seed)
        require(model_cfg["system"] == cfg["system"], "WiMAE physical geometry differs")
        folder = args.training_output / "checkpoints" / name
        ck, evidence = check_training(folder, recipe, model_cfg, lambda p: load_checkpoint(p, "cpu"))
        # The archived cfg retains wimae_ce solely for generator identity. Never
        # route these weights through the historical supervised model registry.
        model = WiMAESliceCE(model_cfg, device=device).to(device).eval()
        model.load_state_dict(ck["model"], strict=True)
        models.append(model)
        configs.append(model_cfg)
        checkpoint_hashes[str(seed)] = evidence["model.pt"]
        training_artifacts[str(seed)] = evidence
        del ck
    payload = dict(routes=routes, cells=cells, suite="manuscript",
                   reference_evaluation_signature=old_ev["signature"],
                   reference_experiment_signature=experiment["signature"],
                   fixed_test_reference=reference.provenance(), training_claim=CLAIM,
                   training_recipe_sha256=file_hash(recipe_path), training_recipe_signature=recipe["signature"],
                   training_artifacts_sha256=training_artifacts, checkpoint_sha256=checkpoint_hashes,
                   implementation_sha256=recipe["implementation_sha256"],
                   scoring_sources_sha256={name: file_hash(Path(__file__).parent / name)
                                          for name in ("evaluate_wimae.py", "wimae_protocol.py")},
                   forti_evaluation_signature=forti_ev["signature"], forti_archive_sha256=forti_zip_sha,
                   forti_delivery=forti_proof, historical_columns=15, reused_forti_columns=3,
                   original_CE_recipe_reproduction=False, samples_per_cell=512, batch_size=8,
                   inference_microbatch=args.inference_microbatch, inference_precision="fp32", test_seed=TEST_SEED,
                   receiver_protocols=["common_diagonal", "native_own_ici_else_diagonal"])
    signature = digest(payload)
    output = args.output.resolve()
    require(output != reference_root and reference_root not in output.parents
            and output != args.forti_evaluation.resolve() and args.forti_evaluation.resolve() not in output.parents,
            "Use a separate WiMAE score output")
    output.mkdir(parents=True, exist_ok=True)
    with P.writer_lock(output, ".wimae_evaluation.lock"):
        metadata = dict(payload, signature=signature)
        if (output / "evaluation.json").exists():
            require(read_json(output / "evaluation.json") == metadata, "Scoring identity changed; use another output")
        write_json(output / "evaluation.json", metadata)
        generator = BatchChannelGenerator(cfg, device, TEST_SEED)
        for number, cell in enumerate(cells, 1):
            check_deadline(deadline)
            path = output / "cells" / (cell["cell_id"] + ".npz")
            marker_path = path.with_suffix(".COMPLETE.json")
            old_path = reference_root / "cells" / path.name
            forti_path = args.forti_evaluation / "cells" / path.name
            old_sha, forti_sha = file_hash(old_path), file_hash(forti_path)
            require(forti_sha == forti_archive_hashes["evaluation/cells/" + path.name]
                    and file_hash(forti_path.with_suffix(".COMPLETE.json")) == forti_archive_hashes[
                        "evaluation/cells/" + cell["cell_id"] + ".COMPLETE.json"],
                    "Forti score/marker changed since verified ZIP binding")
            forti_done = read_json(forti_path.with_suffix(".COMPLETE.json"))
            require(forti_done.get("signature") == forti_ev["signature"]
                    and forti_done.get("source_score_sha256") == old_sha and forti_done.get("score_sha256") == forti_sha,
                    "FortiTran source score changed after verification")
            if path.exists() and marker_path.exists():
                saved = read_json(marker_path)
                require(saved.get("signature") == signature and saved.get("source_score_sha256") == old_sha
                        and saved.get("forti_score_sha256") == forti_sha and saved.get("score_sha256") == file_hash(path),
                        "Stale or damaged saved WiMAE cell")
                with np.load(path, allow_pickle=False) as z:
                    require(z["signature"].item() == signature and z["nmse"].shape == (21, 512)
                            and json.loads(z["meta"].item()) == cell
                            and np.array_equal(z["cluster_id"], np.arange(512) // 8), "Invalid saved WiMAE score layout")
                    reference.assert_fingerprint(cell, z["realization_sha256"].item())
                continue
            started = time.perf_counter()
            with np.load(old_path, allow_pickle=False) as old, np.load(forti_path, allow_pickle=False) as forti:
                require(old["signature"].item() == old_ev["signature"] and forti["signature"].item() == forti_ev["signature"]
                        and json.loads(old["meta"].item()) == json.loads(forti["meta"].item()) == cell
                        and forti["source_score_sha256"].item() == old_sha, "Source score cell binding differs")
                for data in (old, forti):
                    require(np.array_equal(data["cluster_id"], np.arange(512) // 8), "Source cluster layout differs")
                    reference.assert_fingerprint(cell, data["realization_sha256"].item())
                require(forti_done.get("realization_sha256") == old["realization_sha256"].item(),
                        "FortiTran completion waveform differs")
                arrays = {"nmse": np.empty((21, 512), dtype=np.float64), "native_available": np.zeros(21, dtype=np.int8)}
                arrays["nmse"][:15] = old["nmse"][old_indices]
                arrays["nmse"][15:18] = forti["nmse"][forti_indices]
                arrays["native_available"][:15] = old["native_available"][old_indices]
                arrays["native_available"][15:18] = forti["native_available"][forti_indices]
                expected_bits = None
                if cell["ber"]:
                    for key in ("errors_diagonal", "errors_native", "bits", "native_residual_max", "native_direct_fallback",
                                "errors_noise_minus3db", "errors_noise_plus3db"):
                        if key not in old:
                            require(key not in forti, "Source optional receiver fields differ")
                            continue
                        require(key in forti, "FortiTran lacks a receiver field")
                        arrays[key] = np.zeros((21, 512), dtype=old[key].dtype)
                        arrays[key][:15] = old[key][old_indices]
                        arrays[key][15:18] = forti[key][forti_indices]
                    expected_bits = old["bits"][old_indices[0]].copy()
                    require(np.all(arrays["bits"][:18] == expected_bits), "Source BER denominators differ")
                require(np.isfinite(arrays["nmse"][:18]).all(), "Nonfinite reused NMSE")
            waveform = hashlib.sha256()
            with torch.inference_mode():
                for start in range(0, 512, 8):
                    check_deadline(deadline)
                    seed = TEST_SEED + int(digest(dict(cell=cell["cell_id"], offset=start))[:15], 16)
                    rng = torch.Generator(device=device).manual_seed(seed)
                    cond = dict(channel_type=cell["channel"], modulation=cell["modulation"],
                                delay_spread_ns=cell["delay_spread_ns"], doppler_hz=cell["doppler_hz"],
                                cp_len=cell["cp_len"], pilot_spacing=cell["pilot_spacing"])
                    batch = generator.simulate(8, cond, torch.full((8,), cell["snr_db"], device=device), rng, seed + 1000000001)
                    update_waveform_digest(waveform, batch)
                    # Generation and hashing remain8; inference may split only
                    # the received observation tensor, after the frozen draw.
                    for offset in range(0, 8, args.inference_microbatch):
                        check_deadline(deadline)
                        stop = min(offset + args.inference_microbatch, 8)
                        frame_slice = slice(start + offset, start + stop)
                        target = select_data_csi(batch["target_h"][offset:stop], cfg)
                        for index, (model, model_cfg) in enumerate(zip(models, configs), 18):
                            prediction = select_data_csi(model(batch["input"][offset:stop]), model_cfg)
                            values = nmse_per_sample(prediction, target).cpu().numpy()
                            require(values.shape == (stop - offset,) and np.isfinite(values).all(), "Nonfinite or invalid WiMAE NMSE")
                            arrays["nmse"][index, frame_slice] = values
                            if cell["ber"]:
                                symbols, _ = linear_multiuser_equalize(prediction, batch["rx_data"][offset:stop],
                                                                      batch["noise_var"][offset:stop], method="mmse")
                                require(bool(torch.isfinite(symbols).all()), "Nonfinite WiMAE receiver symbols")
                                errors, bits = frame_bit_counts(symbols, batch["tx_label"][offset:stop], cell["modulation"])
                                require(np.array_equal(bits, expected_bits[frame_slice]), "Frozen BER denominator differs")
                                arrays["errors_diagonal"][index, frame_slice] = errors
                                arrays["errors_native"][index, frame_slice] = errors
                                arrays["bits"][index, frame_slice] = bits
                                for db, key in ((-3, "errors_noise_minus3db"), (3, "errors_noise_plus3db")):
                                    if key in arrays:
                                        changed, _ = linear_multiuser_equalize(prediction, batch["rx_data"][offset:stop],
                                            batch["noise_var"][offset:stop] * 10**(db / 10), method="mmse")
                                        require(bool(torch.isfinite(changed).all()), "Nonfinite noise sensitivity symbols")
                                        arrays[key][index, frame_slice] = frame_bit_counts(changed, batch["tx_label"][offset:stop], cell["modulation"])[0]
                            del prediction
                        del target
                    del batch
            reference.assert_fingerprint(cell, waveform.hexdigest())
            check_deadline(deadline)
            arrays.update(cluster_id=np.arange(512) // 8, signature=np.array(signature), meta=np.array(json.dumps(cell)),
                          realization_sha256=np.array(waveform.hexdigest()), source_score_sha256=np.array(old_sha),
                          forti_score_sha256=np.array(forti_sha))
            P.atomic_npz(path, arrays)
            write_json(marker_path, dict(signature=signature, source_score_sha256=old_sha,
                       forti_score_sha256=forti_sha, score_sha256=file_hash(path), realization_sha256=waveform.hexdigest()))
            print(f"[{number}/269] WiMAE3seed frozen SHA PASS {cell['cell_id']}; {time.perf_counter() - started:.1f}s", flush=True)
        check_deadline(deadline)
        write_json(output / "evaluation_complete.json", dict(signature=signature, cells=269))
        aggregate(output, args.bootstrap)
        check_deadline(deadline)
        write_json(output / "COMPLETE.json", dict(status="complete", signature=signature, cells=269,
                   checkpoint_sha256=checkpoint_hashes, paper_ready=False, pending=["paper_assets"]))


if __name__ == "__main__":
    try:
        main()
    except DeadlineReached as error:
        print("WIMAE_EVALUATION_INCOMPLETE_TIME_LIMIT " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(75)
