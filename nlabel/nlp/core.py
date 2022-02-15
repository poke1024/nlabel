import collections
import numpy as np
import orjson
from cached_property import cached_property

from nlabel.io.guid import tagger_guid
from nlabel.io.common import text_hash_code


try:
    import cupy as cp

    def is_cupy_array(arr):
        return isinstance(arr, cp.ndarray)
except ImportError:
    def is_cupy_array(arr):
        return False


try:
    import torch

    def is_torch_tensor(arr):
        return torch.is_tensor(arr)
except ImportError:
    def is_torch_tensor(arr):
        return False


def labels_from_data(data, split=None):
    if data is None or not data:
        return []
    elif split:
        return [{'value': x} for x in sorted(data.split(split))]
    else:
        return [{
            'value': data
        }]


class TagBuilder:
    def __init__(self, name, taggers, save_vectors=None):
        self._name = name
        self._taggers = taggers
        self._save_vectors = save_vectors
        self._vectors = [] if save_vectors else None

    def __len__(self):
        return len(self._taggers)

    def append(self, data, vector=None):
        self._taggers.append(data)

        if self._vectors is not None:
            assert vector is not None
            v = vector()

            if is_cupy_array(v):
                # explicitly convert arrays to numpy arrays.
                v = v.get()
            elif is_torch_tensor(v):
                v = v.detach().cpu().numpy()

            if v.size == 0:
                raise RuntimeError(
                    f"expected vector for tag {self._name}, got {v}")

            self._vectors.append(v)

    def done(self):
        if self._save_vectors and self._vectors:
            self._save_vectors(np.array(self._vectors))


class Builder:
    def __init__(self, guid, signature, taggers=None, vectors: dict = None, renames=None):
        renames = renames if renames else {}
        self._taggers_data = dict(
            (renames.get(name, name), []) for name in taggers) if taggers else {}
        self._renames = renames
        self._vectors = dict((renames.get(k, k), v) for k, v in vectors.items()) if vectors else dict()
        self._vectors_data = {}
        self._data = {
            'guid': guid,
            'tagger': signature,
            'tags': self._taggers_data
        }

    @property
    def data(self):
        return self._data

    @property
    def vectors_data(self):
        return self._vectors_data

    def tagger(self, name, force_empty=True):
        external_name = self._renames.get(name, name)
        tagger = self._taggers_data.get(external_name)
        if tagger is None:
            tagger = []
            self._taggers_data[external_name] = tagger

        if external_name in self._vectors:
            def save_vectors(v):
                self._vectors_data[external_name] = v
        else:
            save_vectors = None

        if force_empty and len(tagger) > 0:
            raise RuntimeError(f"tagger '{name}' not empty")

        return TagBuilder(name, tagger, save_vectors)


class Tagger:
    def __init__(self):
        self._guid = tagger_guid()

    @property
    def guid(self):
        return self._guid

    @staticmethod
    def _env_data():
        import platform
        import nlabel.version

        return {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'runtime': {
                'python': platform.python_version(),
                'nlabel': nlabel.version.__version__
            }
        }

    @property
    def signature(self):
        raise NotImplementedError()

    def process(self, text):
        raise NotImplementedError()

    def process_n(self, texts, batch_size=1):
        return (self.process(x) for x in texts)


class Text(collections.namedtuple('Text', ['text', 'external_key', 'meta'])):
    @cached_property
    def text_hash_code(self):
        return text_hash_code(self.text)

    @cached_property
    def meta_json(self):
        return orjson.dumps(
            self.meta,
            option=orjson.OPT_SORT_KEYS).decode("utf8") if self.meta else ''
