import contextlib
import datetime
import encodings
import gzip
import inspect
import logging
import lzma
import re
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
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
    if "+" in mode:
        raise Exception("open_file does not support mixed reads and writes")
    elif "b" in mode:
        wrap = lambda x: x
    elif mode[0] == "r":
        wrap = encodings.utf_8.StreamReader
        mode = mode[0] + "b"
    elif mode[0] in "wax":
        wrap = encodings.utf_8.StreamWriter
        mode = mode[0] + "b"
    else:
        raise Exception(f"Unsupported mode: {mode!r}")

    if path.suffix == ".zstd":
        return wrap(pyzstd.ZstdFile(path, mode=mode, level_or_option=level))
    elif path.suffix == ".xz":
        return wrap(lzma.LZMAFile(path, mode=mode, preset=level))
    elif path.suffix == ".gz":
        if level is None:
            level = 9
        return wrap(gzip.GzipFile(path, mode=mode, compresslevel=level))
    else:
        return wrap(open(path, mode=mode))


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


def jsonize(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: jsonize(v) for k, v in d.items()}
    elif isinstance(d, (list, tuple)):
        return [jsonize(v) for v in d]
    elif isinstance(d, (int, float, bool, str)) or d is None:
        return d
    elif isinstance(d, np.integer):
        return int(d)
    elif isinstance(d, np.floating):
        return float(d)
    else:
        raise TypeError(f"Unable to JSONize type {type(d)} of {d!r}")


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


FLOAT_RE = r"[-+]?(?:[1-9][0-9]*[.][0-9]*|[.][0-9]+|0[.][0-9]+|[1-9][0-9]*|0)(?:[eE][-+]?[0-9]+)?|[+-]?[iI][nN][fF]|[+-]?[nN][aA][nN]"
INT_RE = r"[+-]?(?:[1-9][0-9]*|0)"


def parse_generator(s: str, seed=None):
    """
    Parse list of numbers or a range to be drawn from.
    Returns a callback generating the next item.
    Ex.: "4.2" "1,2,3" "1.0, 2.3," "2..5" "U(1.0,2)" "LU(1, 1000)"
    """
    s = s.strip().lower()
    r = np.random.RandomState(seed)

    # int, float or mixed value(s)
    m = re.match(rf"^({FLOAT_RE})(?:\s*[,]\s*({FLOAT_RE}))*$", s)
    if m:
        a0 = [x.strip() for x in s.split(",")]
        try:
            a = np.array([int(x) for x in a0])
        except ValueError:
            a = np.array([float(x) for x in a0])
        return lambda: r.choice(a)

    # int range (includive .. exclusive)
    m = re.match(rf"^({INT_RE})\s*..\s*({INT_RE})$", s)
    if m:
        a = [int(x) for x in m.groups()]
        assert len(a) == 2
        return lambda: r.randint(a[0], a[1])

    # uniform float range
    m = re.match(rf"^(l?[u])\s*\(\s*({FLOAT_RE})\s*,\s*({FLOAT_RE})\s*\)$", s)
    if m:
        t = m.groups()[0]
        a = [float(x) for x in m.groups()[1:]]
        assert len(a) == 2
        if t == "u":
            return lambda: r.uniform(a[0], a[1])
        if t == "lu":
            return lambda: np.exp(r.uniform(np.log(a[0]), np.log(a[1])))

    raise ValueError(f"{s!r} can't be parsed as a generator")
