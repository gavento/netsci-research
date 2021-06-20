import contextlib
import datetime
import gzip
import inspect
import logging
import lzma
import time
from pathlib import Path
from typing import Any, Tuple

import pyzstd

log = logging.getLogger(__name__)


def now_isofmt() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_basic_path(path: Path, suffix: str) -> Path:
    """
    Return the path stripped of a known compress suffix and then of `suffix`.

    Raises ValueError if path does not end in `suffix` after stripping compression
    suffix (if present).
    """
    SUFFIXES = [".zstd", ".xz", ".gz"]
    if path.suffix in SUFFIXES:
        path = path.with_suffix()
    if path.suffix != suffix:
        raise ValueError(
            f"{path!r} does not have suffix {suffix!r} (unknown compression?)"
        )
    return path.with_suffix("")


def open_file(path: Path, mode="r", level=None) -> Any:
    if path.suffix == ".zstd":
        return pyzstd.ZstdFile(path, mode=mode, level_or_option=level)
    elif path.suffix == ".xz":
        return lzma.LZMAFile(path, mode=mode, level_or_option=level)
    elif path.suffix == ".gz":
        return gzip.GzipFile(path, mode=mode, compresslevel=level)
    else:
        return open(path, mode=mode)


def get_caller_logger(name="log", levels=2, *, frame=None):
    if frame is None:
        frame = inspect.currentframe()
    l = None
    if levels > 0 and frame.f_back:
        l = get_caller_logger(name=name, levels=levels - 1, frame=frame.f_back)
    if not l:
        if name in frame.f_globals:
            l = frame.f_globals[name]
    return l


@contextlib.contextmanager
def logged_time(name, level=logging.INFO, logger=None):
    """
    Context manager to measure and log operation time.
    """
    if logger is None:
        logger = get_caller_logger()
    t0 = time.time()
    yield
    t1 = time.time()
    logger.log(level, f"{name} took {t1-t0:.3g} s")
