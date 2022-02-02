from .loader import Loader, Document
from ..selector import select_taggers
from cached_property import cached_property

import itertools
import json
import contextlib
import yaml


def split_data(data):
    json_data = dict((k, v) for k, v in data.items() if k != 'vectors')
    vectors_data = data.get('vectors')
    return json_data, vectors_data


def _distinct(values):
    if len(set(map(id, values))) == 1:
        return values[:1]
    return set(map(json.dumps, values))


class Tag:
    def __init__(self, tagger, name):
        self._tagger = tagger
        self._name = name

    @property
    def tagger(self):
        return self._tagger

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"'{self._name}'"


class TaggerPrivate:
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    @staticmethod
    def from_meta(data):
        return Tagger({
            'tagger': data['tagger'],
            'tags': dict((k, None) for k in data['tags'])
        })

    def as_meta(self):
        return {
            'tagger': self._data['tagger'],
            'tags': list(self._data['tags'].keys())
        }


class Tagger:
    def __init__(self, data):
        self._data = data

    @property
    def id(self):
        return self._data['guid']

    @property
    def signature(self):
        return self._data['tagger']

    @cached_property
    def tags(self):
        return [Tag(self, k) for k in self._data['tags'].keys()]

    def __iter__(self):
        for x in self.tags:
            yield x

    def __getattr__(self, k):
        if k not in self._data['tags']:
            raise KeyError(k)
        return Tag(self, k)

    def __str__(self):
        return yaml.dump(self.signature)

    @property
    def _(self):
        return TaggerPrivate(self._data)


class TaggerList(list):
    def __init__(self, taggers):
        super().__init__(taggers)

    def __getitem__(self, x):
        if isinstance(x, dict):
            r = list(select_taggers(self, x))
            if not r:
                raise KeyError(x)
            elif len(r) > 1:
                raise KeyError(x)
            return r[0]
        else:
            return super().__getitem__(x)

    def filter(self, selector):
        return TaggerList(select_taggers(self, selector))


class Group:
    def __init__(self, data):
        if not data.get('vectors'):
            data['vectors'] = [{}] * len(data['taggers'])

        self._data = data

    @staticmethod
    @contextlib.contextmanager
    def open(path, vectors=True):
        from nlabel.io.bahia.single import open_collection
        with open_collection(path, vectors=vectors) as doc:
            yield doc

    def save(self, path, exist_ok=False):
        from nlabel.io.bahia.single import save_doc
        save_doc(self, path, exist_ok=exist_ok)

    @property
    def data(self):
        return self._data

    @property
    def text(self):
        return self._data['text']

    @property
    def meta(self):
        return self._data.get('meta')

    @property
    def external_key(self):
        return self._data.get('external_key')

    @cached_property
    def taggers(self):
        return TaggerList([Tagger(x) for x in self._data['taggers']])

    @property
    def vectors(self):
        r = {}
        v = self._data.get('vectors')
        if v is not None:
            for i, x in enumerate(v):
                if x:
                    r[i] = x
        return r

    def view(self, *selectors, **kwargs):
        loader = Loader(*selectors, **kwargs)
        return loader(self)

    def split(self):
        data = self._data
        if len(data['taggers']) <= 1:
            return [self]

        base_keys = set(data.keys()) - {'taggers', 'vectors'}
        base = dict((k, data[k]) for k in base_keys)

        for nlp, vec in zip(data['taggers'], data['vectors']):
            split_data = base.copy()
            split_data['taggers'] = [nlp]
            split_data['vectors'] = [vec]
            yield Group(split_data)

    @staticmethod
    def join(docs):
        docs = [x.collection if isinstance(x, Document) else x for x in docs]

        if len(docs) == 1:
            return docs[0]

        data = [x.data for x in docs]

        keys = []
        for x in data:
            keys.extend(list(x.keys()))
        keys = set(keys) - {'taggers', 'vectors', 'stat'}

        shared_values = {}
        for k in keys:
            values = [x for x in [x.get(k) for x in data] if x is not None]
            if len(_distinct(values)) > 1:
                raise RuntimeError(
                    f"inconsistent values on key '{k}': {values}")
            if values:
                shared_values[k] = values[0]

        combined = shared_values.copy()
        combined['taggers'] = list(itertools.chain(*[x['taggers'] for x in data]))
        combined['vectors'] = list(itertools.chain(*[x['vectors'] for x in data]))

        return Group(combined)

    def save_to_qda(self, path, *selectors, exist_ok=False):
        from .qda import Exporter as QDAExporter

        exporter = QDAExporter(path, *selectors, exist_ok=exist_ok)
        with exporter.writer() as writer:
            writer.add(self)
