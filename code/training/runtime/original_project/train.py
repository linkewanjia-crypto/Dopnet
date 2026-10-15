import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

from utils.cuda_memory import configure_cuda_allocator

# Configure the CUDA allocator before torch initializes CUDA.  The normal
# training command therefore needs no allocator environment prefix.
configure_cuda_allocator()

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from data.generator import BatchChannelGenerator
from data.channels import validate_channel_config
from data.frame_feasibility import assert_all_feasible, worst_case_ici_to_signal_db
from models.receiver import (
    SharedUserChannelEstimator, ProjectMultiUserReceiver, select_data_csi,
)
from models.baselines import CLASSICAL_BASELINES, TRAINABLE_BASELINES
from utils.metrics import nmse_linear, nmse_per_sample, nmse_summary_db
from utils.checkpoints import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def _deep_merge(base, override):
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_cfg(path):
    """Load YAML with optional ``base_config`` inheritance for ablations."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base_config", None)
    if base is None:
        return cfg
    base_path = (path.parent / base).resolve() if not Path(base).is_absolute() else Path(base)
    return _deep_merge(load_cfg(base_path), cfg)


LEGACY_TRAIN_KEYS = {
    # A fixed-grid MSE on the fused ADD coefficients fights the
    # per-voxel off-grid refinement instead of supervising it.
    "add_fused_loss_weight",
}


def assert_current_config(cfg):
    legacy = sorted(LEGACY_TRAIN_KEYS.intersection(cfg.get("train", {})))
    if legacy:
        raise ValueError(
            "Legacy train configuration keys are unsupported: " + ", ".join(legacy)
        )
    return cfg


def set_nested(cfg, dotted_key, value):
    cur = cfg
    parts = str(dotted_key).split(".")
    if not parts or any(not p for p in parts):
        raise ValueError(f"Invalid --set key: {dotted_key!r}")
    for key in parts[:-1]:
        nxt = cur.get(key)
        if nxt is None:
            nxt = {}
            cur[key] = nxt
        if not isinstance(nxt, dict):
            raise ValueError(f"Cannot set {dotted_key!r}: {key!r} is not a mapping.")
        cur = nxt
    cur[parts[-1]] = value


def parse_set_override(text):
    if "=" not in text:
        raise ValueError(f"--set expects KEY=VALUE, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"--set expects a non-empty key: {text!r}")
    return key, yaml.safe_load(raw)


def resolve_device(name):
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# GPU backend setup and mixed precision
# ---------------------------------------------------------------------------


AMP_DTYPES = {"none": None, "off": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def setup_backend(cfg, device):
    """Enable the throughput features that matter on Ada/Ampere-class GPUs."""
    tr = cfg.get("train", {})
    if device.type != "cuda":
        return {"tf32": False, "cudnn_benchmark": False}
    deterministic = os.environ.get("DOPNET_DETERMINISTIC_TEST") == "1"
    if deterministic:
        # Regression-test mode only. The parent sets CUBLAS_WORKSPACE_CONFIG
        # before this interpreter imports torch.
        torch.use_deterministic_algorithms(True)
    tf32 = bool(tr.get("tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = (
        False if deterministic else bool(tr.get("cudnn_benchmark", True)))
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    return {"tf32": tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_test": deterministic}


def amp_context(cfg, device, enabled=True):
    name = str(cfg.get("train", {}).get("amp", "bf16")).lower()
    dtype = AMP_DTYPES.get(name)
    if dtype is None or device.type != "cuda" or not enabled:
        return contextlib.nullcontext(), None
    return torch.autocast(device_type="cuda", dtype=dtype), dtype


# ---------------------------------------------------------------------------
# optimizer / scheduler
# ---------------------------------------------------------------------------


def build_optimizer(model, train_cfg):
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    betas = tuple(float(x) for x in train_cfg.get("betas", [0.9, 0.999]))
    eps = float(train_cfg.get("eps", 1e-8))
    momentum = float(train_cfg.get("momentum", 0.9))
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(
            "The selected estimator has no trainable parameters. Classical "
            "baselines are evaluation-only; run evaluate.py instead of train.py."
        )
    fused_requested = bool(train_cfg.get("fused_optimizer", False))
    fused_ok = fused_requested and all(p.is_cuda for p in params)
    if name in {"adamw", "adam"}:
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        common = dict(
            lr=lr, weight_decay=wd, betas=betas, eps=eps,
            amsgrad=bool(train_cfg.get("amsgrad", False)),
        )
        if fused_ok:
            try:
                return cls(params, **common, fused=True)
            except (TypeError, RuntimeError) as e:
                warnings.warn(
                    f"Fused {name} unavailable on this build/device; falling back to foreach: {e}"
                )
        return cls(params, **common, foreach=True)
    if name == "radam":
        return torch.optim.RAdam(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    if name == "nadam":
        return torch.optim.NAdam(params, lr=lr, weight_decay=wd, betas=betas, eps=eps,
                                 momentum_decay=float(train_cfg.get("momentum_decay", 4e-3)))
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum,
                               nesterov=bool(train_cfg.get("nesterov", True)))
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd, momentum=momentum,
                                   alpha=float(train_cfg.get("alpha", 0.99)), eps=eps)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=wd, eps=eps)
    raise ValueError(
        f"Unknown optimizer {name!r}. Choose one of: adamw, adam, radam, nadam, "
        "sgd, rmsprop, adagrad."
    )


class ProjectLRScheduler:
    """Serializable epoch-level learning-rate scheduler with optional warm-up."""

    SUPPORTED = {"none", "cosine_warmup", "cosine", "step", "multistep",
                 "exponential", "plateau", "cosine_restarts"}

    def __init__(self, optimizer, train_cfg, total_epochs):
        self.optimizer = optimizer
        self.name = str(train_cfg.get("scheduler", "cosine")).lower()
        if self.name not in self.SUPPORTED:
            raise ValueError(
                f"Unknown scheduler {self.name!r}. Choose one of: "
                + ", ".join(sorted(self.SUPPORTED))
            )
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = max(0, int(train_cfg.get("warmup_epochs", 0)))
        if self.name == "cosine_warmup" and self.warmup_epochs == 0:
            self.warmup_epochs = 5
        self.warmup_start_factor = float(train_cfg.get("warmup_start_factor", 0.1))
        if not (0.0 < self.warmup_start_factor <= 1.0):
            raise ValueError("warmup_start_factor must be in (0,1].")
        self.min_lr = max(0.0, float(train_cfg.get("min_lr", 0.0)))
        self.step_size = max(1, int(train_cfg.get("scheduler_step_size", 15)))
        self.gamma = float(train_cfg.get("scheduler_gamma", 0.5))
        self.milestones = sorted(int(x) for x in train_cfg.get("scheduler_milestones", [20, 40, 50]))
        self.plateau_patience = max(0, int(train_cfg.get("plateau_patience", 5)))
        self.plateau_factor = float(train_cfg.get("plateau_factor", 0.5))
        self.restart_t0 = max(1, int(train_cfg.get("cosine_restart_t0", 10)))
        self.restart_t_mult = max(1, int(train_cfg.get("cosine_restart_t_mult", 2)))

        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        self.current_epoch = 0
        self.best_metric = float("inf")
        self.bad_epochs = 0
        self.plateau_scale = 1.0
        self._set_for_epoch(1)

    def _warmup_factor(self, epoch):
        if self.warmup_epochs <= 0 or epoch > self.warmup_epochs:
            return 1.0
        if self.warmup_epochs == 1:
            return 1.0
        t = (epoch - 1) / float(self.warmup_epochs - 1)
        return self.warmup_start_factor + (1.0 - self.warmup_start_factor) * t

    def _cosine_factor(self, epoch):
        start = self.warmup_epochs + 1
        if epoch <= start:
            return 1.0
        span = max(1, self.total_epochs - start)
        p = min(1.0, max(0.0, (epoch - start) / float(span)))
        return 0.5 * (1.0 + math.cos(math.pi * p))

    def _restart_factor(self, epoch):
        elapsed = max(0, epoch - self.warmup_epochs - 1)
        length = self.restart_t0
        while elapsed >= length:
            elapsed -= length
            length *= self.restart_t_mult
        p = elapsed / float(max(1, length))
        return 0.5 * (1.0 + math.cos(math.pi * p))

    def _factor(self, epoch):
        wf = self._warmup_factor(epoch)
        if epoch <= self.warmup_epochs:
            return wf
        if self.name == "none":
            return 1.0
        if self.name in ("cosine", "cosine_warmup"):
            return self._cosine_factor(epoch)
        if self.name == "step":
            k = max(0, (epoch - self.warmup_epochs - 1) // self.step_size)
            return self.gamma ** k
        if self.name == "multistep":
            k = sum(1 for m in self.milestones if epoch > m)
            return self.gamma ** k
        if self.name == "exponential":
            k = max(0, epoch - self.warmup_epochs - 1)
            return self.gamma ** k
        if self.name == "plateau":
            return self.plateau_scale
        if self.name == "cosine_restarts":
            return self._restart_factor(epoch)
        raise AssertionError(self.name)

    def _set_for_epoch(self, epoch):
        factor = self._factor(int(epoch))
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            if int(epoch) <= self.warmup_epochs:
                lr = base * factor
            else:
                floor_factor = self.min_lr / max(base, 1e-30)
                lr = base * max(floor_factor, factor)
            g["lr"] = float(lr)

    def get_last_lr(self):
        return [float(g["lr"]) for g in self.optimizer.param_groups]

    def step(self, metric=None):
        completed_epoch = self.current_epoch + 1
        if self.name == "plateau" and completed_epoch > self.warmup_epochs:
            value = float(metric) if metric is not None else float("inf")
            if value < self.best_metric - 1e-12:
                self.best_metric = value
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                if self.bad_epochs > self.plateau_patience:
                    self.plateau_scale *= self.plateau_factor
                    self.bad_epochs = 0
        self.current_epoch = completed_epoch
        next_epoch = min(self.total_epochs, self.current_epoch + 1)
        self._set_for_epoch(next_epoch)

    def state_dict(self):
        return {"name": self.name, "total_epochs": self.total_epochs,
                "current_epoch": self.current_epoch, "best_metric": self.best_metric,
                "bad_epochs": self.bad_epochs, "plateau_scale": self.plateau_scale,
                "base_lrs": list(self.base_lrs)}

    def load_state_dict(self, state):
        self.current_epoch = int(state["current_epoch"])
        self.best_metric = float(state["best_metric"])
        self.bad_epochs = int(state["bad_epochs"])
        self.plateau_scale = float(state["plateau_scale"])
        if len(state["base_lrs"]) != len(self.base_lrs):
            raise RuntimeError("Checkpoint scheduler parameter-group count mismatch")
        self.base_lrs = [float(x) for x in state["base_lrs"]]
        next_epoch = min(self.total_epochs, self.current_epoch + 1)
        self._set_for_epoch(max(1, next_epoch))


def build_scheduler(optimizer, train_cfg, total_epochs):
    return ProjectLRScheduler(optimizer, train_cfg, total_epochs)


# ---------------------------------------------------------------------------
# losses and stepping
# ---------------------------------------------------------------------------


def move_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def model_inputs(batch):
    """Only the keys that an estimator is allowed to see."""
    return {"add_input": batch.get("add_input"),
            "add_input_slow": batch.get("add_input_slow"),
            "add_input_pilot_only_fast_time": batch.get("add_input_pilot_only_fast_time"),
            "noise_var": batch.get("noise_var"),
            "ls_grid": batch.get("ls_grid")}


def dopnet_auxiliary_loss(aux, batch, cfg):
    """Physics-guided auxiliary objectives (proposed model only).

    The fixed-grid target supervises only the pre-refinement ADD denoiser, which
    is well posed because ``add_branch_pred`` lives on the fixed grid and is
    produced before any refinement.  The fused representation is supervised
    through the unitary physics synthesis instead, which keeps coefficients and
    per-voxel frequencies consistent.
    """
    if not isinstance(aux, dict) or aux.get("architecture") != "dual3d_physics_fusion":
        return batch["input"].new_zeros(()), {}
    tr = cfg.get("train", {})
    total = batch["input"].new_zeros(())
    stats = {}

    # A zero branch weight is a true target-free ablation: do not even read
    # the simulator-derived coefficient label for loss or monitoring.
    if (float(tr.get("add_branch_loss_weight", 0.0)) > 0.0
            and "target_add" in batch and "add_branch_pred" in aux):
        target = batch["target_add"]
        branch = aux["add_branch_pred"]
        branch_mse = F.mse_loss(branch, target)
        total = total + float(tr["add_branch_loss_weight"]) * branch_mse
        stats["add_branch_mse"] = branch_mse.detach()
        if "add_fused_pred" in aux:
            # Monitoring only: the fused coefficients are evaluated at refined
            # frequencies, so a fixed-grid MSE on them is not the objective.
            stats["add_fused_mse_monitor"] = F.mse_loss(
                aux["add_fused_pred"].detach(), target
            ).detach()

    # This regularizer does not use the simulator-derived coefficient target,
    # so retain it in the no_add_aux control.
    if "add_fused_pred" in aux:
        fused = aux["add_fused_pred"]
        mag = torch.sqrt(fused[:, :, 0].pow(2) + fused[:, :, 1].pow(2) + 1e-12)
        sparse = mag.mean()
        total = total + float(tr["add_sparsity_weight"]) * sparse
        stats["add_sparsity"] = sparse.detach()

    if "physics_h" in aux:
        ph = select_data_csi(aux["physics_h"], cfg)
        th = select_data_csi(batch["target_h"], cfg)
        phys_mse = F.mse_loss(ph, th)
        total = total + float(tr["physics_cfr_loss_weight"]) * phys_mse
        stats["physics_cfr_mse"] = phys_mse.detach()

    if "doppler_offset_bins" in aux:
        off = aux["doppler_offset_bins"]
        total = total + float(tr["doppler_offset_l2_weight"]) * off.pow(2).mean()
        stats["doppler_offset_rms_bins"] = off.pow(2).mean().sqrt().detach()
        if "doppler_offset_hz" in aux:
            stats["doppler_offset_rms_hz"] = aux["doppler_offset_hz"].pow(2).mean().sqrt().detach()
    return total, stats


def compute_loss(model, b, cfg, task):
    kw = model_inputs(b)
    if task == "joint":
        joint_kw = {k: v for k, v in kw.items() if k != 'noise_var'}
        hhat, zri, aux = model(b["input"], b["rx_data"], b["noise_var"],
                               b["modulation_id"], return_aux=True, **joint_kw)
    else:
        hhat, aux = model(b["input"], return_aux=True, **kw)
    hhat_score = select_data_csi(hhat.float(), cfg)
    h_score = select_data_csi(b["target_h"], cfg)
    # Keep the public compute_loss signature unchanged while allowing the
    # trainable comparison models to use their published loss family by default.
    # train.baseline_loss_mode="unified" restores the previous common MSE+NMSE loss.
    estimator = str(cfg.get("model", {}).get("estimator", "")).lower()
    baseline_loss_mode = str(cfg.get("train", {}).get("baseline_loss_mode", "paper")).lower()
    if estimator in TRAINABLE_BASELINES and baseline_loss_mode == "paper":
        if estimator == "ha02":
            loss = F.smooth_l1_loss(hhat_score, h_score, beta=1.0)
        else:
            loss = F.mse_loss(hhat_score, h_score)
    elif baseline_loss_mode not in {"paper", "unified"}:
        raise ValueError(
            f"Unknown train.baseline_loss_mode={baseline_loss_mode!r}; "
            "choose 'paper' or 'unified'.")
    else:
        loss = F.mse_loss(hhat_score, h_score) \
            + float(cfg["train"]["nmse_weight"]) * nmse_linear(hhat_score, h_score)
    if task == "joint":
        loss = loss + float(cfg["train"]["symbol_loss_weight"]) * F.mse_loss(zri.float(), b["tx_symbol"])
    aux_loss, aux_stats = dopnet_auxiliary_loss(aux, b, cfg)
    return loss + aux_loss, hhat_score, h_score, aux, aux_stats


@torch.no_grad()
def validate_batches(model, batches, device, task, cfg):
    """Validate in full precision with the canonical project NMSE."""
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    nmse_values = []
    bs = int(cfg["train"]["batch_size"])
    for start in range(0, batches["input"].shape[0], bs):
        b = move_batch({k: v[start:start + bs] for k, v in batches.items()}, device)
        loss, hhat_score, h_score, aux, _ = compute_loss(model, b, cfg, task)
        n = int(b["input"].shape[0])
        loss_sum += float(loss) * n
        sample_count += n
        nmse_values.extend(nmse_per_sample(hhat_score, h_score).detach().cpu().numpy())
    out = {"loss": loss_sum / max(sample_count, 1),
           "nmse_db": nmse_summary_db(nmse_values)[0]}
    return out


# ---------------------------------------------------------------------------
# validation cache
# ---------------------------------------------------------------------------


def validation_cache_path(cfg, seed):
    m = cfg.get("model", {})
    payload = {
        "cache_schema": "dopnet_common_observations_v1",
        "generator_source": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((Path(__file__).parent / 'data').glob('*.py'))},
        "system": cfg["system"],
        "val_samples": int(cfg["train"]["val_samples"]),
        "seed": int(seed),
        "generator": cfg.get("data_pipeline", {}),
        # Generator outputs depend on these observation-grid settings.
        "observation": {k: m.get(k) for k in
                        ("add_doppler_bins", "doppler_whitening_reg",
                         "add_delay_bins", "add_delay_guard_bins")},
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return Path("cache") / f"validation_{h}.pt"


def load_or_build_validation(cfg, device, seed, fixed_cache=None, fixed_sha256=None):
    if fixed_cache is not None:
        from fixed_validation import load_historical_validation
        return load_historical_validation(cfg, fixed_cache, fixed_sha256)
    seed = int(cfg['train'].get('validation_seed', seed))
    need_pilot_only = (
        str(cfg.get('model', {}).get('ablation_profile', 'full')).lower()
        == 'pilot_only_fast_time'
        or bool(cfg.get('data_pipeline', {}).get('emit_pilot_only_fast_time_add', False))
    )
    cfg = _deep_merge(cfg, {'model': {'ablation_profile': 'full'},
                           'data_pipeline': {'emit_pilot_ls': True,
                                             'emit_slow_time_add': True,
                                             'emit_pilot_only_fast_time_add': need_pilot_only}})
    use_cache = bool(cfg["train"].get("validation_cache", True))
    path = validation_cache_path(cfg, seed)
    if use_cache and path.exists():
        print(f"[validation-cache] HIT {path}")
        return load_checkpoint(path, map_location="cpu")
    print(f"[validation-cache] MISS {path}")
    gen = BatchChannelGenerator(cfg, device=device, seed=seed + 900001)
    data = gen.materialize_cpu(int(cfg["train"]["val_samples"]), epoch=0,
                               batch_size=int(cfg["train"]["batch_size"]))
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(data, path)
        print(f"[validation-cache] SAVED {path}")
    return data


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dopnet.yaml")
    ap.add_argument("--run-name", default="full")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--init-checkpoint", default=None,
                    help="Warm-start model weights only; optimizer is reinitialized.")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-samples", type=int, default=None)
    ap.add_argument("--val-samples", type=int, default=None)
    ap.add_argument("--generation-batch-size", type=int, default=None)
    ap.add_argument("--microbatch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--estimator", default=None,
                    help="Override model.estimator (dopnet or a baseline name).")
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default=None)
    ap.add_argument("--optimizer",
                    choices=["adamw", "adam", "radam", "nadam", "sgd", "rmsprop", "adagrad"],
                    default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--nmse-weight", type=float, default=None)
    ap.add_argument("--symbol-loss-weight", type=float, default=None)
    ap.add_argument("--betas", type=float, nargs=2, default=None, metavar=("BETA1", "BETA2"))
    ap.add_argument("--momentum", type=float, default=None)
    ap.add_argument("--scheduler",
                    choices=["none", "cosine_warmup", "cosine", "step", "multistep",
                             "exponential", "plateau", "cosine_restarts"], default=None)
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--min-lr", type=float, default=None)
    ap.add_argument("--scheduler-step-size", type=int, default=None)
    ap.add_argument("--scheduler-gamma", type=float, default=None)
    ap.add_argument("--set", dest="set_values", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="Generic dotted YAML override, repeatable.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument('--epoch-cache', default=None)
    ap.add_argument('--fixed-validation-cache', default=None,
                    help='Use the exact archived common validation tensors; never regenerate them.')
    ap.add_argument('--fixed-validation-sha256', default=None,
                    help='Required SHA-256 of --fixed-validation-cache.')
    ap.add_argument('--stop-after-epoch', type=int, default=None,
                    help='Stop at this epoch WITHOUT changing scheduler horizon')
    return ap


def apply_overrides(cfg, args):
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.train_samples is not None:
        cfg["train"]["train_samples"] = int(args.train_samples)
    if args.val_samples is not None:
        cfg["train"]["val_samples"] = int(args.val_samples)
    if args.generation_batch_size is not None:
        cfg.setdefault("data_pipeline", {})["generation_batch_size"] = int(args.generation_batch_size)
    if args.microbatch_size is not None:
        cfg.setdefault("data_pipeline", {})["microbatch_size"] = int(args.microbatch_size)
    if args.device is not None:
        cfg["device"] = args.device
    if args.estimator is not None:
        cfg["model"]["estimator"] = args.estimator
    if args.amp is not None:
        cfg["train"]["amp"] = args.amp
    if args.optimizer is not None:
        cfg["train"]["optimizer"] = args.optimizer
    if args.lr is not None:
        cfg["train"]["lr"] = float(args.lr)
    if args.weight_decay is not None:
        cfg["train"]["weight_decay"] = float(args.weight_decay)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.grad_clip is not None:
        cfg["train"]["grad_clip"] = float(args.grad_clip)
    if args.nmse_weight is not None:
        cfg["train"]["nmse_weight"] = float(args.nmse_weight)
    if args.symbol_loss_weight is not None:
        cfg["train"]["symbol_loss_weight"] = float(args.symbol_loss_weight)
    if args.betas is not None:
        cfg["train"]["betas"] = [float(x) for x in args.betas]
    if args.momentum is not None:
        cfg["train"]["momentum"] = float(args.momentum)
    if args.scheduler is not None:
        cfg["train"]["scheduler"] = args.scheduler
    if args.warmup_epochs is not None:
        cfg["train"]["warmup_epochs"] = int(args.warmup_epochs)
    if args.min_lr is not None:
        cfg["train"]["min_lr"] = float(args.min_lr)
    if args.scheduler_step_size is not None:
        cfg["train"]["scheduler_step_size"] = int(args.scheduler_step_size)
    if args.scheduler_gamma is not None:
        cfg["train"]["scheduler_gamma"] = float(args.scheduler_gamma)
    for item in args.set_values:
        key, value = parse_set_override(item)
        set_nested(cfg, key, value)
    return cfg


def main():
    args = build_argparser().parse_args()
    cfg = assert_current_config(load_cfg(args.config))
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    cfg = apply_overrides(cfg, args)
    cfg["train"].setdefault("optimizer", "adamw")
    cfg["train"].setdefault("scheduler", "cosine")

    estimator = str(cfg.get("model", {}).get("estimator", "dopnet")).lower()
    if estimator in CLASSICAL_BASELINES:
        raise SystemExit(
            f"'{estimator}' is a closed-form baseline with no trainable parameters. "
            "Evaluate it directly:\n"
            f"  python evaluate.py --config {args.config} --estimator {estimator}"
        )
    if str(cfg["model"].get("ablation_profile", "full")) == "no_fast_time_add":
        cfg.setdefault("data_pipeline", {})["emit_slow_time_add"] = True
    if str(cfg["model"].get("ablation_profile", "full")) == "pilot_only_fast_time":
        cfg.setdefault("data_pipeline", {})["emit_pilot_only_fast_time_add"] = True

    validate_channel_config(cfg["system"])
    assert_all_feasible(cfg)
    if args.smoke:
        cfg["train"]["epochs"] = 2
        cfg["train"]["train_samples"] = 64
        cfg["train"]["val_samples"] = 32
        cfg["data_pipeline"]["generation_batch_size"] = min(
            32, int(cfg["data_pipeline"]["generation_batch_size"]))
        cfg["data_pipeline"]["microbatch_size"] = min(
            8, int(cfg["data_pipeline"]["microbatch_size"]))

    cfg["seed"] = seed
    if bool(args.fixed_validation_cache) != bool(args.fixed_validation_sha256):
        raise ValueError(
            '--fixed-validation-cache and --fixed-validation-sha256 must be given together'
        )
    if args.fixed_validation_cache:
        from fixed_validation import assert_historical_identity, _expected_cache_hash
        # Check the scientific configuration before model allocation.  The
        # expensive file hash and safe tensor load happen once at first use.
        assert_historical_identity(cfg)
        _expected_cache_hash(args.fixed_validation_sha256)
    if args.epoch_cache:
        if args.stop_after_epoch is None:
            raise ValueError('--epoch-cache requires --stop-after-epoch')
        from epoch_cache import check_identity
        # Reject mismatches before model allocation or costly validation data.
        check_identity(args.epoch_cache,cfg,args.stop_after_epoch)
    device = resolve_device(cfg.get("device", "auto"))
    backend = setup_backend(cfg, device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    task = str(cfg["train"].get("task", "ce")).lower()
    model = (ProjectMultiUserReceiver(cfg) if task == "joint"
             else SharedUserChannelEstimator(cfg)).to(device)
    # channels_last_3d is only valid for 5-D Conv3d weights.  Published DL
    # baselines are faithful 2-D/attention architectures, so forcing this
    # memory format onto them raises on Conv2d weights and is not meaningful.
    use_channels_last_3d = (
        bool(cfg["train"].get("channels_last_3d", False))
        and device.type == "cuda"
        and estimator not in TRAINABLE_BASELINES
    )
    if use_channels_last_3d:
        model = model.to(memory_format=torch.channels_last_3d)

    profile = str(cfg.get("model", {}).get("ablation_profile", "full"))
    print("device:", device, "backend:", backend)
    print("estimator:", estimator, "| ablation_profile:", profile)
    print("amp:", cfg["train"].get("amp", "bf16"),
          "| grad_checkpoint:", cfg["model"].get("grad_checkpoint", False))
    print("parameters:", count_parameters(model))
    print("training channels:", cfg["system"]["channel_types"])
    print("channel weights:", cfg["system"].get("channel_type_weights"))
    print("data pipeline:", cfg.get("data_pipeline", {}))
    print(f"worst-case ICI/signal: {worst_case_ici_to_signal_db(cfg):.2f} dB "
          f"vs SNR range [{cfg['system']['snr_db_min']}, {cfg['system']['snr_db_max']}] dB")

    start_epoch = 1
    best = float("inf")
    optimizer_state = scheduler_state = None
    resume_optimizer_name = resume_scheduler_name = None
    if args.resume:
        ck = load_checkpoint(args.resume, "cpu")
        if ck.get('validation_cache_sha256') != (
                args.fixed_validation_sha256.lower()
                if args.fixed_validation_sha256 else None):
            raise RuntimeError(
                'Resume validation cache SHA-256 differs; use the original fixed cache.'
            )
        model.load_state_dict(ck["model"], strict=True)
        optimizer_state = ck["optimizer"]
        scheduler_state = ck["scheduler"]
        resume_scheduler_name = str(ck["scheduler_name"]).lower()
        resume_optimizer_name = str(ck["optimizer_name"]).lower()
        start_epoch = int(ck["epoch"]) + 1
        if ck.get('selection_metric') != 'val_nmse_db':
            raise RuntimeError('Resume requires the new NMSE-selected checkpoint format.')
        if ck['cfg'] != cfg:
            raise RuntimeError('Resume configuration differs; use a new run directory.')
        best = float(ck['best_val_nmse_db'])
        print("resumed:", args.resume, "from epoch", start_epoch - 1)
    elif args.init_checkpoint:
        ck = load_checkpoint(args.init_checkpoint, "cpu")
        source_task = str(ck["task"]).lower()
        source_state = ck["model"]
        if task == "joint" and source_task != "joint":
            source_state = {("estimator." + k if k.startswith("base.") else k): v
                            for k, v in source_state.items()}
        elif task != "joint" and source_task == "joint":
            source_state = {k[len("estimator."):]: v for k, v in source_state.items()
                            if k.startswith("estimator.")}
        missing, unexpected = model.load_state_dict(source_state, strict=False)
        allowed_missing = ([k for k in missing if k.startswith("detector.")]
                           if task == "joint" and source_task != "joint" else [])
        real_missing = [k for k in missing if k not in allowed_missing]
        if real_missing or unexpected:
            raise RuntimeError(
                f"Warm-start incompatibility. missing={real_missing}, unexpected={unexpected}")
        print("warm-started model weights from:", args.init_checkpoint)

    optimizer_name = str(cfg["train"].get("optimizer", "adamw")).lower()
    opt = build_optimizer(model, cfg["train"])
    if optimizer_state is not None:
        if resume_optimizer_name != optimizer_name:
            raise RuntimeError(
                f"Checkpoint optimizer {resume_optimizer_name!r} != current {optimizer_name!r}")
        opt.load_state_dict(optimizer_state)
    print("optimizer:", optimizer_name, opt.defaults)

    scheduler_name = str(cfg["train"].get("scheduler", "cosine")).lower()
    scheduler = build_scheduler(opt, cfg["train"], int(cfg["train"]["epochs"]))
    if args.resume:
        if resume_scheduler_name != scheduler_name:
            raise RuntimeError(
                f"Checkpoint scheduler {resume_scheduler_name!r} != current {scheduler_name!r}")
        scheduler.load_state_dict(scheduler_state)
    print("scheduler:", scheduler_name, "warmup_epochs=",
          cfg["train"].get("warmup_epochs", 0), "initial_lr=", scheduler.get_last_lr())

    run_dir = Path("checkpoints") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    val_data = load_or_build_validation(
        cfg, device, seed, args.fixed_validation_cache, args.fixed_validation_sha256,
    )
    train_gen = BatchChannelGenerator(cfg, device=device, seed=seed + 10001)
    history = []
    if (run_dir / "history.csv").exists() and args.resume:
        import pandas as pd
        history = pd.read_csv(run_dir / "history.csv").to_dict("records")
        history = [r for r in history if int(r['epoch']) < start_epoch]

    amp_name = str(cfg["train"].get("amp", "bf16")).lower()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_name == "fp16" and device.type == "cuda"))
    if args.resume:
        scaler.load_state_dict(ck['scaler'])
        random.setstate(ck['rng_python'])
        np.random.set_state(ck['rng_numpy'])
        torch.set_rng_state(ck['rng_torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(ck['rng_cuda'])
    total_epochs = int(cfg["train"]["epochs"])
    epoch_cache_hash = None
    if args.epoch_cache:
        from experiment_protocol import file_hash
        epoch_cache_hash = file_hash(Path(args.epoch_cache)/'complete.json')
    stop_epoch = total_epochs if args.stop_after_epoch is None else min(total_epochs, args.stop_after_epoch)
    if args.epoch_cache and start_epoch != stop_epoch:
        raise ValueError('One shared cache must cover exactly the next epoch')
    for epoch in range(start_epoch, stop_epoch + 1):
        model.train()
        epoch_lr = float(opt.param_groups[0]["lr"])
        running = []
        t0 = time.perf_counter()
        n_seen = 0
        batches = train_gen.iter_batches(int(cfg["train"]["train_samples"]), epoch,
                                         int(cfg["train"]["batch_size"]))
        if args.epoch_cache:
            from epoch_cache import read_batches
            batches = read_batches(args.epoch_cache, cfg, epoch, device)
        total_steps = math.ceil(int(cfg["train"]["train_samples"])
                                / int(cfg["train"]["batch_size"]))
        # Always show the live training bar. run_experiments.py forwards the child
        # process as a raw byte stream, preserving tqdm's carriage-return redraws
        # instead of translating each redraw into a new log line.
        bar = tqdm(
            batches,
            total=total_steps,
            desc=f"epoch {epoch}/{stop_epoch}",
            file=sys.stdout,
            disable=False,
            dynamic_ncols=False,
            mininterval=0.5,
            leave=False,
        )
        for b in bar:
            n_seen += int(b["input"].shape[0])
            opt.zero_grad(set_to_none=True)
            ctx, _ = amp_context(cfg, device)
            with ctx:
                loss, _, _, _, _ = compute_loss(model, b, cfg, task)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]), error_if_nonfinite=True)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]), error_if_nonfinite=True)
                opt.step()
            running.append(float(loss.detach()))
            bar.set_postfix(loss=f"{np.mean(running[-20:]):.4f}")

        train_seconds = time.perf_counter() - t0
        val = validate_batches(model, val_data, device, task, cfg)
        scheduler.step(val["loss"])
        row = {
            "epoch": epoch, "lr": epoch_lr,
            "next_lr": float(opt.param_groups[0]["lr"]),
            "train_loss": float(np.mean(running)),
            "val_loss": val["loss"], "val_nmse_db": val["nmse_db"],
            "train_seconds": train_seconds,
            "train_samples_per_s": n_seen / max(train_seconds, 1e-9),
            "peak_memory_mb": (torch.cuda.max_memory_allocated() / 1024 ** 2
                               if device.type == "cuda" else float("nan")),
        }
        history.append(row)
        print(
            f"epoch {epoch}/{stop_epoch} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_nmse_db={row['val_nmse_db']:.3f} | "
            f"lr={row['lr']:.3e} | "
            f"samples/s={row['train_samples_per_s']:.1f} | "
            f"peak_mem_mb={row['peak_memory_mb']:.0f}",
            flush=True,
        )
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerows(history)

        state = {
            "model": model.state_dict(), "optimizer": opt.state_dict(), "cfg": cfg,
            "validation_cache_sha256": (
                args.fixed_validation_sha256.lower()
                if args.fixed_validation_sha256 else None
            ),
            "epoch_data_manifest_sha256": epoch_cache_hash,
            "optimizer_name": optimizer_name, "optimizer_config": dict(cfg["train"]),
            "scheduler": scheduler.state_dict(), "scheduler_name": scheduler_name,
            "scheduler_config": {k: cfg["train"].get(k) for k in (
                "scheduler", "warmup_epochs", "warmup_start_factor", "min_lr",
                "scheduler_step_size", "scheduler_gamma", "scheduler_milestones",
                "plateau_patience", "plateau_factor",
                "cosine_restart_t0", "cosine_restart_t_mult")},
            "task": task, "estimator": estimator, "ablation_profile": profile,
            "epoch": epoch, "best_val_loss": val['loss'], "val": val,
            "selection_metric": "val_nmse_db",
            "best_val_nmse_db": min(best, val['nmse_db']),
            "scaler": scaler.state_dict(),
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            "trainable_parameters": count_parameters(model),
            "data_pipeline": cfg.get("data_pipeline", {}),
            "scientific_conventions": {
                "snr_reference": cfg["system"].get("snr_reference"),
                "snr_noise_policy": cfg["system"].get("snr_noise_policy"),
                "target_reference": cfg["system"].get("target_reference"),
                "waveform_fft_size": cfg["system"].get("waveform_fft_size"),
                "amp": amp_name,
            },
        }
        if not math.isfinite(val['nmse_db']):
            raise RuntimeError('Non-finite validation NMSE; experiment is invalid.')
        if val["nmse_db"] < best:
            best = val["nmse_db"]
            save_checkpoint(state, run_dir / "model.pt")
            print("saved", run_dir / "model.pt")
        save_checkpoint(state, run_dir / "last.pt")


if __name__ == "__main__":
    main()
