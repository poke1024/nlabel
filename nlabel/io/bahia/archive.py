import contextlib
import functools
import orjson
import h5py
import zipfile

from tqdm.auto import tqdm
from cached_property import cached_property

from nlabel.io.common import save_archive
from nlabel.io.json.archive import Archive as AbstractArchive
from nlabel.io.json.collection import Collection
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
    def __init__(self, path, zf, vf):
        self._path = path
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

                yield stem, Collection(data)

    def _keyed_collections(self, progress):
        external_keys = {}
        for k, indices in self.external_keys.items():
            for i in indices:
                external_keys[i] = k

        for stem, doc in self._collections(progress=progress):
            yield external_keys.get(int(stem)), doc

    def save(self, path, engine, export_opts=None, exist_ok=False, progress=True):
        save_archive(
            path, engine, functools.partial(self._keyed_collections, progress=progress),
            export_opts=export_opts, exist_ok=exist_ok)

    def __len__(self):
        return self._size

    def iter(self, *selectors, progress=True):
        loader = Loader(*selectors)
        for _, doc in self._collections(progress=progress):
            yield loader(doc)

    def iter_c(self, progress=True):
        for _, collection in self._collections(progress=progress):
            yield collection


@contextlib.contextmanager
def open_archive(path, mode: str = "r", archive_guid=None):
    if mode != "r":
        raise RuntimeError(f"mode {mode} is not supported")

    with zipfile.ZipFile(path / "documents.zip", "r") as zf:
        vectors_path = path / "vectors.h5"
        if vectors_path.exists():
            with h5py.File(vectors_path, "r") as vf:
                if vf.attrs['archive'] != archive_guid:
                    raise RuntimeError("archive GUID mismatch")
                yield Archive(path, zf, vf)
        else:
            yield Archive(path, zf, None)

