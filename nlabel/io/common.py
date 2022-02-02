import contextlib
import numpy as np
import shutil
import json

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
    from nlabel.io.arriba.generate import make_archive as make_arriba_archive

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


def save_archive(path, engine, taggers, keyed_docs, export_opts=None, exist_ok=False):
    from nlabel.io.arriba.generate import make_archive as make_arriba_archive
    from nlabel.io.bahia.generate import make_archive as make_bahia_archive

    base_path = to_path(path, '.nlabel')

    if base_path.exists():
        if exist_ok:
            shutil.rmtree(base_path)
        else:
            raise RuntimeError(f"{base_path} already exists")

    base_path.mkdir()

    try:
        if not export_opts:
            export_opts = {}

        if engine == 'bahia':
            export_keys = export_opts.get('export_keys', True)
            export_opts = dict((k, v) for k, v in export_opts.items() if k != 'export_keys')

            docs = keyed_docs(**export_opts)
            make_bahia_archive(taggers, docs, base_path, export_keys=export_keys)
            #_profile(make_bahia_archive, docs, base_path, export_keys=export_keys)
        elif engine == 'arriba':
            docs = keyed_docs(**export_opts)
            make_arriba_archive(taggers, docs, base_path)
            #_profile(make_arriba_archive, docs, base_path)
        else:
            raise ValueError(f'unsupported storage engine {engine}')

    except:
        if base_path.exists():
            shutil.rmtree(base_path)
        raise


class ArchiveInfo:
    def __init__(self, path, mode=None, engine=None):
        base_path = to_path(path, '.nlabel')
        auto_mode = False

        if mode is None:
            auto_mode = True
            if base_path.exists():
                mode = 'r+'
            else:
                mode = 'w+'

        if mode == 'r':
            if not base_path.exists():
                raise FileNotFoundError(base_path)
            with open(base_path / 'meta.json', 'r') as f:
                meta = json.loads(f.read())
            if meta['type'] != 'archive':
                raise RuntimeError(
                    f"expected archive, got {meta['type']}")
            if engine is None:
                engine = meta['engine']
            elif engine != meta['engine']:
                raise ValueError(
                    'specified engine does not match archive')
            archive_guid = meta['guid']
        elif mode in ('r+', 'w+'):
            if base_path.exists():
                if mode == 'w+':
                    raise RuntimeError(
                        f"ignoring w+ on existing archive file at {base_path}")

                with open(base_path / 'meta.json', 'r') as f:
                    meta = json.loads(f.read())

                if engine is None:
                    engine = meta['engine']
                elif engine != meta['engine']:
                    raise ValueError(
                        'specified engine does not match archive')

                assert meta['type'] == 'archive'
                assert meta['version'] == 1

                archive_guid = meta['guid']
            else:
                if mode == 'r+':
                    raise FileNotFoundError(base_path)

                if engine is None:
                    raise ValueError('specify a storage engine')

                base_path.mkdir()

                archive_guid = make_archive_guid()

                meta = {
                    'type': 'archive',
                    'engine': engine,
                    'version': 1,
                    'guid': archive_guid,
                    'taggers': []
                }

                with open(base_path / 'meta.json', 'w') as f:
                    f.write(json.dumps(meta))
        else:
            raise ValueError(mode)

        if engine in ('bahia', 'arriba') and auto_mode and mode == 'r+':
            mode = 'r'

        self.base_path = base_path
        self.mode = mode
        self.engine = engine
        self.guid = archive_guid
        self.taggers = meta["taggers"]
        self.meta = meta


@contextlib.contextmanager
def open_archive(path, mode=None, engine=None, **kwargs):
    from nlabel.io.carenero.archive import open_archive as open_carenero_archive
    from nlabel.io.bahia.archive import open_archive as open_bahia_archive
    from nlabel.io.arriba.archive import open_archive as open_arriba_archive

    info = ArchiveInfo(path, mode=mode, engine=engine)

    if info.engine == 'carenero':
        with open_carenero_archive(info, **kwargs) as x:
            yield x
    elif info.engine == 'bahia':
        with open_bahia_archive(info, **kwargs) as x:
            yield x
    elif info.engine == 'arriba':
        with open_arriba_archive(info, **kwargs) as x:
            yield x
    else:
        raise ValueError(f'unknown engine {info.engine}')


class RemoteArchive:
    def __init__(self, api_url, auth=None):
        self.api_url = api_url
        self.auth = auth
