"""Kernel-owned pool locks: concurrent jobs remain exclusive to legacy runners."""
from contextlib import contextmanager, ExitStack
import fcntl
import os


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"a live local writer owns {path.name}") from error
        yield stream


@contextmanager
def pool_locks(root):
    """Hold both legacy locks; children inherit the same open descriptions."""
    directory = root / "artifacts/phase3/locks"
    with ExitStack() as stack:
        locks = {kind: stack.enter_context(exclusive(directory / name)) for kind, name in (
            ("cuda", "native-worker.lock"), ("cpu", "cpu-native-worker.lock"))}
        yield {kind: stream.fileno() for kind, stream in locks.items()}


@contextmanager
def job_lock(root, identity, *, cpu_only, pool_fd=None):
    if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("invalid job identity")
    directory = root / "artifacts/phase3/locks"
    path = directory / ("cpu-native-worker.lock" if cpu_only else "native-worker.lock")
    with ExitStack() as stack:
        if pool_fd is None:
            stack.enter_context(exclusive(path))
        else:
            # A numeric/environment token is not a lock. Require a descriptor
            # for this exact inode and a real exclusive kernel lock. Never
            # unlock it here: the controller and siblings share its ownership.
            info, expected = os.fstat(pool_fd), path.stat()
            if (info.st_dev, info.st_ino) != (expected.st_dev, expected.st_ino):
                raise ValueError("pool descriptor does not own the resource lock")
            fcntl.flock(pool_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stack.enter_context(exclusive(directory / "jobs" / f"{identity}.lock"))
        yield
