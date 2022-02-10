from nlabel.io.common import text_hash_code
from .loader import Loader, Document
from ..selector import select_taggers, auto_selectors
from ..guid import text_guid
from .name import Name

from cached_property import cached_property
from collections import Counter
import collections

import itertools
import orjson
import contextlib
import yaml


def split_data(data):
    json_data = dict((k, v) for k, v in data.items() if k != 'vectors')
    vectors_data = data.get('vectors')
    return json_data, vectors_data


def _distinct(values):
    if len(set(map(id, values))) == 1:
        return values[:1]
    return set(map(orjson.dumps, values))


class StrLabelFactory:
    def make_empty_label(self, kind):
        if kind in ('label', 'default'):
            return ''
        elif kind == 'labels':
            return []
        else:
            raise ValueError(kind)

    def make_label(self, data, kind):
        if kind in ('label', 'default'):
            return data
        elif kind == 'labels':
            return [data]
        else:
            raise ValueError(kind)


class VoidLabelFactory:
    def make_empty_label(self, kind):
        raise None

    def make_label(self, data, kind):
        return None


class TransformedLabelFactory:
    def __init__(self, base, tfms):
        self._base = base
        self._tfms = tfms

    def make_empty_label(self, kind):
        label = self._base.make_empty_label(kind)
        for f in self._tfms:
            label = f(label)
        return label

    def make_label(self, data, kind):
        label = self._base.make_label(data, kind)
        for f in self._tfms:
            label = f(label)
        return label


class ListLabelFactory:
    def __init__(self, elem):
        self._elem = elem

    def make_empty_label(self, kind):
        if kind == 'label':
            raise ValueError(
                "cannot access list as 'label'")
        elif kind in ('labels', 'default'):
            return []
        else:
            raise ValueError(kind)

    def make_label(self, data, kind):
        if kind == 'label':
            raise ValueError(
                "cannot access list as 'label'")
        elif kind in ('labels', 'default'):
            f = self._elem
            return [f.make_label(x, 'label') for x in data]
        else:
            raise ValueError(kind)


class StructureLabelFactory:
    def __init__(self, members):
        self._members = members
        self._type = collections.namedtuple(
            'Structure', members)

    def make_empty_label(self, kind):
        if kind == 'label':
            return None
        elif kind in ('labels', 'default'):
            return []
        else:
            raise ValueError(kind)

    def make_label(self, data, kind):
        if kind in ('label', 'default'):
            return self._type(**data)
        elif kind == 'labels':
            return [self._type(**data)]
        else:
            raise ValueError(kind)


def make_label_factory(grammar):
    if isinstance(grammar, dict):
        gt = grammar['type']
        if gt == 'list':
            return ListLabelFactory(
                make_label_factory(grammar['member']))
        elif gt == 'structure':
            return StructureLabelFactory(
                grammar['members'])
        else:
            raise ValueError(
                f'unimplemented label grammar "{grammar}"')
    elif grammar == 'str':
        return StrLabelFactory()
    elif grammar == 'void':
        return VoidLabelFactory()
    else:
        raise ValueError(
            f'illegal label grammar "{grammar}"')


class DataType:
    def __init__(self, data):
        self._data = data

    @property
    def json(self):
        return self._data

    def __str__(self):
        return yaml.dump(self._data)


class Tag:
    def __init__(self, tagger, name: Name, tfms=None):
        self._tagger = tagger
        self._name = name
        self._tfms = tfms or []

    def transform(self, f):
        return Tag(self._tagger, self._name, self._tfms + [f])

    def to(self, name: str = None):
        if name is None:
            new_name = self._name
        else:
            new_name = Name(self._name.internal, name)
        return Tag(self._tagger, new_name, self._tfms)

    @property
    def tagger(self):
        return self._tagger

    @property
    def name(self):
        return self._name.external

    @cached_property
    def dtype(self):
        return DataType(self._tagger.signature['grammar'].get(
            self._name.internal))

    @property
    def _label_factory(self):
        f = make_label_factory(self.dtype.json)
        if self._tfms:
            f = TransformedLabelFactory(f, self._tfms)
        return f

    def __str__(self):
        return self._name.external

    def __repr__(self):
        return f"'{self._name.external}'"


class TaggerPrivate:
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    @staticmethod
    def from_meta(data):
        return Tagger({
            'guid': data['guid'],
            'tagger': data['signature'],
            'tags': dict((k, None) for k in data['tags'])
        })

    def as_meta(self):
        return {
            'guid': self._data['guid'],
            'signature': self._data['tagger'],
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
        return [Tag(self, Name(k)) for k in self._data['tags'].keys()]

    def __iter__(self):
        for x in self.tags:
            yield x

    def __getattr__(self, k):
        if k not in self._data['tags']:
            raise KeyError(k)
        return Tag(self, Name(k))

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
        root = self._data['root']
        if root['type'] == 'text':
            return root['text']
        else:
            raise ValueError(
                f"cannot get text from root '{root['type']}'")

    @cached_property
    def text_hash_code(self):
        return text_hash_code(self.text)

    @property
    def meta(self):
        return self._data.get('meta')

    @cached_property
    def meta_json(self):
        return orjson.dumps(
            self.meta,
            option=orjson.OPT_SORT_KEYS).decode("utf8") if self.meta else ''

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
        selectors = auto_selectors(selectors, self.taggers)
        loader = Loader(*selectors, **kwargs)
        return loader(self)

    def split(self):
        data = self._data
        if len(data['taggers']) <= 1:
            yield self
        else:
            base_keys = set(data.keys()) - {'taggers', 'vectors', 'guid'}
            base = dict((k, data[k]) for k in base_keys)

            for nlp, vec in zip(data['taggers'], data['vectors']):
                split_data = base.copy()
                split_data['guid'] = text_guid()
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
        keys = set(keys) - {'taggers', 'vectors', 'stat', 'guid'}

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

        tagger_guids = Counter([x['guid'] for x in combined['taggers']])
        if any(x > 1 for x in tagger_guids.values()):
            raise RuntimeError("cannot join due to duplicate tagger GUIDs")

        combined['guid'] = text_guid()

        return Group(combined)

    def save_to_qda(self, path, *selectors, exist_ok=False):
        from .qda import Exporter as QDAExporter

        exporter = QDAExporter(path, *selectors, exist_ok=exist_ok)
        with exporter.writer() as writer:
            writer.add(self)
