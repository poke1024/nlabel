from nlabel.io.common import open_archive, RemoteArchive
from nlabel.io.slice import Slice
from nlabel.io.carenero.mt import batch_add as batch_add_mt
from typing import Union, Any
from nlabel.nlp import NLP
from typing import List

import contextlib


@contextlib.contextmanager
def _open_archive(path, base_path):
    if path is None:
        path = base_path.parent / (base_path.stem + ".nlabel")

    if path.exists():
        mode = 'r+'
    else:
        mode = 'w+'

    with open_archive(path, mode=mode, engine="carenero") as archive:
        yield archive


class Importer:
    def __init__(self, nlp: Union[NLP, Any], base_path):
        if not isinstance(nlp, NLP):
            nlp = NLP(nlp)
        self._nlp = nlp
        self._base_path = base_path

    def _items(self):
        raise NotImplementedError()

    def to_local_archive(self, path=None):
        with _open_archive(path, self._base_path) as archive:
            archive.batch_add(self._nlp, self._items())

    def to_remote_archive(self, archive: RemoteArchive, batch_size=16):
        batch_add_mt(archive, self._nlp, self._items(), batch_size=batch_size)

    def to_minstrel(self):
        raise NotImplementedError()


class Selection:
    def __init__(self, filters=None):
        self._filters = filters if filters else []

    @property
    def filters(self):
        return self._filters

    def by_key(self, rows: List[Union[str, dict]]):
        return Selection(self._filters + [('by_key', rows)])

    def by_index(self, rows: List[int]):
        return Selection(self._filters + [('by_index', rows)])

    def by_slice(self, slice):
        return Selection(self._filters + [('by_slice', slice)])
