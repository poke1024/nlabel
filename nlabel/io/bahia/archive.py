import contextlib
import orjson
import h5py
import zipfile

from tqdm.auto import tqdm
from cached_property import cached_property

from nlabel.io.common import make_writer
from nlabel.io.selector import auto_selectors
from nlabel.io.json.archive import Archive as AbstractArchive
from nlabel.io.json.group import Group, TaggerPrivate as JsonTagger, TaggerList as JsonTaggerList
from nlabel.io.json.loader import Loader


class VectorsData:
    def __init__(self, v_data):
        self._v_data = v_data

    def __getitem__(self, i):
        return self._v_data[str(i)]

    def get(self, i):
        return self._v_data.get(str(i))

    def __len__(self):
        return len(self._v_data.keys())

    def __iter__(self):
        keys = sorted(map(int, self._v_data.keys()))
        for k in keys:
            yield self._v_data[str(k)]


class Archive(AbstractArchive):
    def __init__(self, path, taggers, zf, vf):
        self._path = path
        self._taggers = taggers
        self._zf = zf
        self._vf = vf
        self._size = len(zf.filelist)

    @property
    def path(self):
        return self._path

    @cached_property
    def external_keys(self):
        has_keys = True
        p = "/keys.json"
        try:
            self._zf.getinfo(p)
        except KeyError:
            has_keys = False
        if has_keys:
            return orjson.loads(self._zf.readstr(p)).items()
        else:
            return {}

    def _collections(self, progress=False, **kwargs):
        assert not kwargs

        zf = self._zf
        vf = self._vf

        for name in tqdm(
                zf.namelist(),
                total=self._size,
                disable=not progress):

            stem = name.split('.')[0].strip()
            if stem.isdigit():
                data = orjson.loads(zf.read(name))
                if vf is not None:
                    v_data = vf.get(stem)
                    if v_data is not None:
                        data['vectors'] = VectorsData(v_data)

                yield stem, Group(data)

    def _groups(self, progress):
        external_keys = {}
        for k, indices in self.external_keys.items():
            for i in indices:
                external_keys[i] = k

        for stem, group in self._collections(progress=progress):
            yield external_keys.get(int(stem)), group

    def save(self, path, engine, options=None, exist_ok=False, progress=True):
        w = make_writer(path, engine, exist_ok=exist_ok)
        options = w.set_options(options)
        if options:
            raise ValueError(f"unsupported options {options}")
        w.write(self._groups(progress=progress), self.taggers)

    def __len__(self):
        return self._size

    @cached_property
    def taggers(self):
        return JsonTaggerList([JsonTagger.from_meta(x) for x in self._taggers])

    def iter_groups(self, progress=True):
        for _, collection in self._collections(progress=progress):
            yield collection

    def iter(self, *selectors, progress=True):
        selectors = auto_selectors(selectors, self.taggers)
        loader = Loader(*selectors)
        for _, doc in self._collections(progress=progress):
            yield loader(doc)


@contextlib.contextmanager
def open_archive(info):
    path = info.base_path
    mode = info.mode

    if mode != "r":
        raise RuntimeError(f"mode {mode} is not supported")

    with zipfile.ZipFile(path / "documents.zip", "r") as zf:
        vectors_path = path / "vectors.h5"
        if vectors_path.exists():
            with h5py.File(vectors_path, "r") as vf:
                if vf.attrs['archive'] != info.guid:
                    raise RuntimeError("archive GUID mismatch")
                yield Archive(path, info.taggers, zf, vf)
        else:
            yield Archive(path, info.taggers, zf, None)

