"""New-run authorization, STOP and resource policy; standard library only.

No legacy deadline, PID, receipt or authorization is accepted by this policy.
Importing it does not query a GPU, read an environment or touch an output.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import uuid


RUN_SCHEMA = "dopnet-lee-research-candidate-run-v1"
RESOURCE_SCHEMA = "dopnet-lee-research-candidate-resources-v1"
AUTHORIZATION_NAME = "RUN_AUTHORIZATION.json"
GIB = 1024 ** 3
MIB = 1024 ** 2


class PolicyError(RuntimeError):
    """The new-run request does not satisfy its own explicit policy."""


class PolicyStopped(PolicyError):
    """A STOP request or the caller's new deadline prevents further work."""


def utc_now():
    return datetime.now(timezone.utc)


def _utc(value):
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise PolicyError("deadline_utc must be an explicit timezone-aware UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise PolicyError("deadline_utc must include a UTC timezone")
    return parsed.astimezone(timezone.utc)


def _path(value, *, existing=False, directory=False):
    path = Path(value).absolute()
    if ".." in path.parts or any(part.lower() in (".aws", ".ssh", ".env") or part.lower().startswith(".env.") for part in path.parts):
        raise PolicyError("unsafe path component or secret-store path")
    for item in reversed((path, *path.parents)):
        if item.is_symlink():
            raise PolicyError("symlink path components are not allowed")
    if existing and not path.exists():
        raise PolicyError("required path does not exist: " + str(path))
    if directory and not path.is_dir():
        raise PolicyError("required directory does not exist: " + str(path))
    return path


def _read_json(path):
    if not path.is_file() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise PolicyError("new-run metadata must be a regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PolicyError("new-run metadata is unreadable") from exc
    if not isinstance(value, dict):
        raise PolicyError("new-run metadata must be an object")
    return value


def _signed(value):
    """Public content-integrity digest; it is not account authentication."""
    if "signature" in value:
        raise PolicyError("new identity must not already contain a signature")
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise PolicyError("new-run metadata is not finite JSON") from exc
    return dict(value, signature=hashlib.sha256(raw).hexdigest())


def _verify_signed(value):
    unsigned = {key: item for key, item in value.items() if key != "signature"}
    if value.get("signature") != _signed(unsigned)["signature"]:
        raise PolicyError("new-run metadata content-integrity signature is invalid")
    return value


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise PolicyError(name + " must be a positive integer")
    return value


def _query(words):
    try:
        result = subprocess.run(words, check=True, capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PolicyError("fresh local NVIDIA resource query failed") from exc
    return result.stdout


def resource_check(*, output, cpu_threads=1, require_cuda=True, require_idle_gpu=True,
                   min_free_host_memory_bytes=GIB, min_free_gpu_memory_mib=38400,
                   min_total_gpu_memory_mib=47000, min_free_disk_bytes=GIB):
    """Inspect actual CPU/RAM/disk and, if requested, one idle NVIDIA GPU.

    CUDA library availability remains the controller's torch.cuda check. This
    standard-library guard measures driver-visible capacity without importing
    torch. No environment is changed and no resource is reserved by this query.
    """
    if os.name != "posix":
        raise PolicyError("the independent candidate currently requires POSIX")
    _positive(cpu_threads, "cpu_threads")
    for value, name in ((min_free_host_memory_bytes, "min_free_host_memory_bytes"),
                        (min_free_disk_bytes, "min_free_disk_bytes")):
        _positive(value, name)
    for value, name in ((min_free_gpu_memory_mib, "min_free_gpu_memory_mib"),
                        (min_total_gpu_memory_mib, "min_total_gpu_memory_mib")):
        if type(value) is not int or value < 0:
            raise PolicyError(name + " must be a nonnegative integer")
    destination = _path(output)
    parent = _path(destination.parent, existing=True, directory=True)
    available_cpu = os.cpu_count()
    if available_cpu is None or cpu_threads > available_cpu:
        raise PolicyError("requested CPU threads exceed known actual capacity")
    try:
        host_free = int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        disk_free = shutil.disk_usage(parent).free
    except (OSError, ValueError, KeyError) as exc:
        raise PolicyError("actual host memory/disk capacity could not be measured") from exc
    if host_free < min_free_host_memory_bytes or disk_free < min_free_disk_bytes:
        raise PolicyError("actual host RAM or free output disk is below the declared resource floor")
    report = dict(schema=RESOURCE_SCHEMA, checked_utc=utc_now().isoformat(),
                  platform="posix", requested_cpu_threads=cpu_threads,
                  available_cpu_threads=available_cpu, free_host_memory_bytes=host_free,
                  free_output_disk_bytes=disk_free, output_parent=str(parent),
                  require_cuda=bool(require_cuda), GPU_queried=False,
                  min_free_host_memory_bytes=min_free_host_memory_bytes,
                  min_free_disk_bytes=min_free_disk_bytes)
    if not require_cuda:
        return report
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise PolicyError("CUDA run requires a working local nvidia-smi executable")
    gpu_text = _query([executable, "--query-gpu=index,uuid,memory.total,memory.free",
                       "--format=csv,noheader,nounits"])
    gpus = []
    for row in csv.reader(io.StringIO(gpu_text), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 4:
            raise PolicyError("NVIDIA capacity output has an unsupported format")
        try:
            index, total, free = int(row[0]), int(row[2]), int(row[3])
        except ValueError as exc:
            raise PolicyError("NVIDIA capacity is not a concrete numeric measurement") from exc
        if total <= 0 or free < 0 or free > total or not row[1].strip():
            raise PolicyError("invalid NVIDIA memory/identity measurement")
        gpus.append(dict(index=index, uuid=row[1].strip(), total_mib=total, free_mib=free))
    # This is the only environment key used; it is a GPU selector, not a secret.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        selectors = [part.strip() for part in visible.split(",") if part.strip()]
        if len(selectors) != 1 or selectors[0] == "-1":
            raise PolicyError("exactly one CUDA device must be visible")
        selected = [gpu for gpu in gpus if selectors[0] in (str(gpu["index"]), gpu["uuid"])]
    else:
        selected = gpus
    if len(selected) != 1:
        raise PolicyError("the new run requires exactly one identifiable visible NVIDIA GPU")
    gpu = selected[0]
    if gpu["total_mib"] < min_total_gpu_memory_mib or gpu["free_mib"] < min_free_gpu_memory_mib:
        raise PolicyError("actual single-GPU total/free memory is below the declared resource floor")
    process_text = _query([executable, "--query-compute-apps=gpu_uuid,pid",
                           "--format=csv,noheader,nounits"])
    other_pids, own_context = [], False
    for row in csv.reader(io.StringIO(process_text), skipinitialspace=True):
        if not row or (len(row) == 1 and row[0].strip() == "No running processes found"):
            continue
        if len(row) != 2:
            raise PolicyError("NVIDIA compute-process output has an unsupported format")
        try:
            pid = int(row[1])
        except ValueError as exc:
            raise PolicyError("NVIDIA process identity is not concrete") from exc
        if row[0].strip() == gpu["uuid"]:
            if pid == os.getpid():
                own_context = True
            else:
                other_pids.append(pid)
    if require_idle_gpu and other_pids:
        raise PolicyError("the selected GPU has other active compute processes")
    total_bytes = gpu["total_mib"] * MIB
    process_limit = total_bytes - 3 * GIB
    if process_limit <= 0:
        raise PolicyError("GPU capacity cannot accommodate the mandatory 3 GiB process reserve")
    report.update(GPU_queried=True, selected_gpu=gpu, other_active_compute_pids=other_pids,
                  own_process_has_GPU_context=own_context, require_idle_gpu=require_idle_gpu,
                  min_free_gpu_memory_mib=min_free_gpu_memory_mib,
                  min_total_gpu_memory_mib=min_total_gpu_memory_mib,
                  cuda_process_memory_limit_bytes=process_limit,
                  cuda_process_memory_fraction=process_limit / total_bytes,
                  gpu_reserved_from_process_bytes=3 * GIB,
                  incremental_estimator_budget_bytes=6 * GIB,
                  torch_cuda_availability_verified_by_policy=False)
    return report


@dataclass
class RunPolicy:
    project: Path | str
    output: Path | str
    authorization_text: str
    acknowledge_new_run: bool
    resume: bool = False
    stop_file: Path | str | None = None
    deadline_utc: str | datetime | None = None
    run_kind: str = "train"
    cpu_threads: int = 1
    min_free_host_memory_bytes: int = GIB
    min_free_gpu_memory_mib: int = 38400
    min_total_gpu_memory_mib: int = 47000
    min_free_disk_bytes: int = GIB
    run_id: str = field(init=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _created_utc: str = field(init=False, repr=False)

    def __post_init__(self):
        if self.acknowledge_new_run is not True:
            raise PolicyError("explicit acknowledge_new_run is required")
        if not isinstance(self.authorization_text, str) or not self.authorization_text.strip():
            raise PolicyError("actual nonempty authorization_text is required")
        if len(self.authorization_text) > 16384:
            raise PolicyError("authorization_text is too long")
        if self.run_kind not in ("prepare", "train", "evaluate"):
            raise PolicyError("unknown independent new-run kind")
        self.project = _path(self.project, existing=True, directory=True)
        self.output = _path(self.output)
        if self.output == self.project or self.output == Path(self.output.anchor):
            raise PolicyError("output must be a distinct named run directory")
        _path(self.output.parent, existing=True, directory=True)
        self.stop_file = _path(self.stop_file) if self.stop_file is not None else None
        self.deadline_utc = _utc(self.deadline_utc) if self.deadline_utc is not None else None
        self.run_id = str(uuid.uuid4())
        self._created_utc = utc_now().isoformat()
        self.check()

    def _resume_record(self):
        record = _verify_signed(_read_json(_path(self.output / AUTHORIZATION_NAME, existing=True)))
        if (record.get("schema") != RUN_SCHEMA or record.get("project") != str(self.project)
                or record.get("output") != str(self.output) or record.get("run_kind") != self.run_kind
                or record.get("acknowledge_new_run") is not True):
            raise PolicyError("resume accepts only this namespace's matching new-run schema")
        try:
            uuid.UUID(record["run_id"])
            _utc(record["authorization_utc"])
        except (KeyError, ValueError, TypeError) as exc:
            raise PolicyError("resume metadata lacks a genuine new-run identity/time") from exc
        return record

    def check(self):
        """Cheap callback for iteration boundaries; it never runs nvidia-smi."""
        if os.name != "posix":
            raise PolicyError("the independent candidate currently requires POSIX")
        if self.deadline_utc is not None and utc_now() >= self.deadline_utc:
            raise PolicyStopped("the operator's explicit new-run deadline has elapsed")
        for path in (self.project / "STOP", self.output / "STOP", self.stop_file):
            if path is not None and (path.exists() or path.is_symlink()):
                raise PolicyStopped("STOP is present: " + str(path))
        _path(self.project, existing=True, directory=True)
        _path(self.output)
        if self.output.exists():
            if not self.output.is_dir():
                raise PolicyError("run output is not a directory")
            if not self.resume and not self._initialized:
                raise PolicyError("fresh run output must not already exist")
            record = self._resume_record()
            if self._initialized and record["run_id"] != self.run_id:
                raise PolicyError("active run identity changed")
        elif self.resume:
            raise PolicyError("resume output does not exist")
        return None

    def as_callback(self):
        return self.check

    def resource_check(self, *, require_cuda=None, require_idle_gpu=True):
        self.check()
        if require_cuda is None:
            require_cuda = self.run_kind != "prepare"
        value = resource_check(output=self.output, cpu_threads=self.cpu_threads,
                               require_cuda=require_cuda, require_idle_gpu=require_idle_gpu,
                               min_free_host_memory_bytes=self.min_free_host_memory_bytes,
                               min_free_gpu_memory_mib=self.min_free_gpu_memory_mib,
                               min_total_gpu_memory_mib=self.min_total_gpu_memory_mib,
                               min_free_disk_bytes=self.min_free_disk_bytes)
        self.check()
        return value

    def initialize_output(self, resource_snapshot=None):
        self.check()
        if self._initialized:
            raise PolicyError("run output has already been initialized by this policy")
        event = dict(schema=RUN_SCHEMA, run_kind=self.run_kind, project=str(self.project),
                     output=str(self.output), acknowledge_new_run=True,
                     authorization_text=self.authorization_text,
                     authorization_utc=utc_now().isoformat(), process_id=os.getpid(),
                     run_id=self.run_id, deadline_utc=self.deadline_utc.isoformat() if self.deadline_utc else None,
                     legacy_authorization_or_receipt_reused=False,
                     supplied_resource_snapshot=resource_snapshot)
        if self.resume:
            record = self._resume_record()
            self.run_id = record["run_id"]
            event.update(run_id=self.run_id, resume=True)
            event = _signed(event)
            events = _path(self.output / "AUTHORIZATION_EVENTS.jsonl")
            if events.exists() and not events.is_file():
                raise PolicyError("authorization events must be a regular file")
            with events.open("a", encoding="utf-8") as dst:
                dst.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            self.output.mkdir(exist_ok=False)
            record = _signed(dict(event, resume=False))
            with (self.output / AUTHORIZATION_NAME).open("x", encoding="utf-8") as dst:
                json.dump(record, dst, ensure_ascii=False, indent=2, sort_keys=True)
                dst.write("\n")
        self._initialized = True
        self.check()
        return record
