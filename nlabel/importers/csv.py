import csv
import mmap
import codecs

from nlabel import NLP, Slice
from nlabel.nlp.core import Text
from .base import Importer as AbstractImporter, Selection
from cached_property import cached_property

from pathlib import Path
from tqdm import tqdm
from typing import List, Union, Any


def _count_data_rows(f):
    lines = 0
    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as mm:
        with codecs.getreader("utf-8")(mm) as text:
            last_pos = 0
            with tqdm(desc="counting lines", total=mm.size(), unit_scale=True) as pbar:
                for _ in csv.reader(iter(text.readline, '')):
                    lines += 1
                    pos = mm.tell()
                    pbar.update(pos - last_pos)
                    last_pos = pos
                if lines > 0:
                    lines -= 1  # exclude header
    return lines


class ByKeySelection:
    def __init__(self, external_keys):
        if not external_keys:
            self._keys = []
            self._select = {}
        else:
            keys = sorted(external_keys[0].keys())
            self._keys = keys

            if not all(sorted(x.keys()) == keys for x in external_keys):
                raise RuntimeError("inhomogenous keys not supported")

            self._select = set(
                tuple([xk[k] for k in keys])
                for xk in external_keys)

    def __call__(self, i, row):
        return tuple([row[k] for k in self._keys]) in self._select


class ByIndexSelection:
    def __init__(self, indices):
        self._indices = set(indices)

    def __call__(self, i, row):
        return i in self._indices


class BySliceSelection:
    def __init__(self, arg):
        self._slice = arg if isinstance(arg, Slice) else Slice(arg)

    def __call__(self, i, row):
        return self._slice(i)


class Filter:
    def __init__(self, fs):
        self._i = [0] * len(fs)
        self._fs = fs

    def __call__(self, row):
        for j, f in enumerate(self._fs):
            ok = f(self._i[j], row)
            self._i[j] += 1
            if not ok:
                return False
        return True


def _make_filter(selection: Selection, keys):
    if selection is None:
        return lambda row: True
    else:
        constructor = {
            'by_key': ByKeySelection,
            'by_index': ByIndexSelection,
            'by_slice': BySliceSelection
        }

        fs = []
        for f_name, f_arg in selection.filters:
            fs.append(constructor[f_name](f_arg))

        return Filter(fs)


class Importer(AbstractImporter):
    def __init__(self, csv_instance, nlp: Union[NLP, Any], keys, text, selection: Selection = None):
        self._csv = csv_instance
        self._keys = sorted(keys)
        self._text_key = text
        self._csv_path = csv_instance.path
        self._filter = _make_filter(selection, keys)
        super().__init__(nlp, self._csv.path)

    def _items(self):
        with open(self._csv_path, "r") as f:
            n_rows = self._csv.num_rows
            reader = csv.DictReader(f)
            for i, row in enumerate(tqdm(reader, total=n_rows, desc=f"processing {self._csv_path}")):
                if not self._filter(row):
                    continue

                if len(self._keys) == 1:
                    external_key = row[self._keys[0]]
                else:
                    external_key = dict((k, row[k]) for k in self._keys)

                text = row[self._text_key]
                meta = dict((k, v) for k, v in row.items() if k != self._text_key)

                yield Text(
                    text=text,
                    external_key=external_key,
                    meta=meta)


class CSV:
    def __init__(self, csv_path: Union[str, Path], keys: List[str], text: str = 'text'):
        self._csv_path = Path(csv_path)
        self._keys = keys
        self._text = text

    @property
    def path(self):
        return self._csv_path

    @cached_property
    def num_rows(self):
        with open(self._csv_path, "r") as f:
            return _count_data_rows(f)

    def importer(self, nlp: Union[NLP, Any], selection: Selection = None):
        return Importer(self, nlp, self._keys, self._text, selection=selection)
