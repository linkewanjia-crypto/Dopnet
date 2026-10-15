"""OS-released lock: a crashed process cannot leave a stale ownership lock."""
import contextlib
from pathlib import Path

@contextlib.contextmanager
def experiment_lock(output):
    import fcntl  # Training server is Linux.
    path=Path(output); path.mkdir(parents=True,exist_ok=True)
    with (path/'.run.lock').open('a') as handle:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process owns this experiment output') from error
        try: yield
        finally: fcntl.flock(handle,fcntl.LOCK_UN)
