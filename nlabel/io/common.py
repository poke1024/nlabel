import contextlib
import numpy as np
import shutil
import orjson
import hashlib

from numpy import searchsorted
from pathlib import Path
from nlabel.io.guid import archive_guid as make_archive_guid


class AbstractSpanFactory:
    def __init__(self):
        self._spans = {}

    def _make_span(self, start, end):
        raise NotImplementedError()

    def from_json(self, data):
        start = data.get("start")
        end = data.get("end")
        if start is not None and end is not None:
            return self.get(start, end)
        else:
            return None

    def get(self, start, end):
        k = (start, end)
        span = self._spans.get(k)
        if span is None:
            span = self._make_span(start, end)
            self._spans[k] = span
        return span

    @property
    def sorted_spans(self):
        return sorted(
            self._spans.values(),
            key=lambda s: (s.start, s.start - s.end))


class TagError(AttributeError):
    def __init__(self, tag):
        super().__init__(f"span has no tag '{tag}'")


def binary_searcher(values, dtype=np.int32):
    arr = np.array(values, dtype=dtype)
    n = len(arr)

    def index(x):
        i = searchsorted(arr, x)
        return i if (i < n and arr[i] == x) else None

    return index


def to_path(p, suffix):
    p = Path(p)
    if p.suffix == '':
        return p.with_suffix(suffix)
    if p.suffix != suffix:
        raise ValueError(f"expected suffix {suffix}, got {p.suffix}")
    return p


def _profile(export, docs, *args, n=100, **kwargs):
    import cProfile
    import itertools
    with cProfile.Profile() as pr:
        export(
            itertools.islice(docs, n), *args, **kwargs)

    import io
    import pstats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
    with open("profile.txt", "w") as f:
        f.write(s.getvalue())


class AbstractWriter:
    def __init__(self, path, exist_ok=False):
        self._path = path
        self._exist_ok = exist_ok

    def set_options(self, options):
        raise NotImplementedError()

    def _write(self, base_path, groups, taggers):
        raise NotImplementedError()

    def write(self, groups, taggers):
        base_path = to_path(self._path, '.nlabel')

        if base_path.exists():
            if self._exist_ok:
                shutil.rmtree(base_path)
            else:
                raise RuntimeError(f"{base_path} already exists")

        base_path.mkdir()

        try:
            self._write(base_path, groups, taggers)
        except:
            if base_path.exists():
                shutil.rmtree(base_path)
            raise


def make_writer(path, engine, exist_ok=False):
    if engine == 'bahia':
        from nlabel.io.bahia.generate import BahiaWriter
        return BahiaWriter(path, exist_ok=exist_ok)
    elif engine == 'arriba':
        from nlabel.io.arriba.generate import ArribaWriter
        return ArribaWriter(path, exist_ok=exist_ok)
    else:
        raise ValueError(f'unsupported storage engine {engine}')


class ArchiveInfo:
    def __init__(self, path, mode=None, engine=None, exist_ok=False):
        base_path = to_path(path, '.nlabel')

        self.base_path = base_path
        self.mode = mode
        self.auto_mode = False
        self.engine = engine
        self.guid = None
        self.meta = None

        self._detect_mode()

        if self.mode in ('r', 'r+'):
            if base_path.exists():
                self._read()
            else:
                raise FileNotFoundError(base_path)
        elif self.mode in ('w', 'w+'):
            if base_path.exists():
                if not exist_ok:
                    raise RuntimeError(
                        f"archive file at {base_path} exists")
                else:
                    shutil.rmtree(base_path)

            self._create()
        else:
            raise ValueError(f"unsupported mode {self.mode}")

        if self.engine in ('bahia', 'arriba') and self.auto_mode and self.mode == 'r+':
            self.mode = 'r'

        assert self.guid is not None
        assert isinstance(self.meta, dict)
        self.taggers = self.meta["taggers"]

    def _detect_mode(self):
        if self.mode is None:
            self.auto_mode = True
            if self.base_path.exists():
                self.mode = 'r+'
            else:
                self.mode = 'w+'

    def _read(self):
        with open(self.base_path / 'meta.json', 'r') as f:
            meta = orjson.loads(f.read())

        if meta['type'] != 'archive':
            raise RuntimeError(
                f"expected archive, got {meta['type']}")

        if meta['version'] != 1:
            raise RuntimeError(
                f"expected version 1, got {meta['version']}")

        if self.engine is None:
            self.engine = meta['engine']
        elif self.engine != meta['engine']:
            raise ValueError(
                'specified engine does not match archive')

        self.meta = meta
        self.guid = meta['guid']

    def _create(self):
        if self.engine is None:
            raise ValueError('specify a storage engine')

        self.base_path.mkdir()

        self.guid = make_archive_guid()

        meta = {
            'type': 'archive',
            'engine': self.engine,
            'version': 1,
            'guid': self.guid,
            'taggers': []
        }

        with open(self.base_path / 'meta.json', 'wb') as f:
            f.write(orjson.dumps(meta))

        self.meta = meta


@contextlib.contextmanager
def open_archive(path, mode=None, engine=None, exist_ok=False, **kwargs):
    info = ArchiveInfo(path, mode=mode, engine=engine, exist_ok=exist_ok)

    if info.engine == 'carenero':
        from nlabel.io.carenero.archive import open_archive as open_carenero_archive
        with open_carenero_archive(info, **kwargs) as x:
            yield x
    elif info.engine == 'bahia':
        from nlabel.io.bahia.archive import open_archive as open_bahia_archive
        with open_bahia_archive(info, **kwargs) as x:
            yield x
    elif info.engine == 'arriba':
        from nlabel.io.arriba.archive import open_archive as open_arriba_archive
        with open_arriba_archive(info, **kwargs) as x:
            yield x
    else:
        raise ValueError(f'unknown engine {info.engine}')


class RemoteArchive:
    def __init__(self, api_url, auth=None):
        self.api_url = api_url
        self.auth = auth


def text_hash_code(text):
    return hashlib.blake2b(
        text.encode("utf8"), digest_size=32).hexdigest()

