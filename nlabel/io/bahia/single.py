import orjson
import h5py
import contextlib

from nlabel.io.json.group import split_data, Group
from nlabel.io.common import to_path
from nlabel.io.guid import archive_guid as make_archive_guid


@contextlib.contextmanager
def open_collection(path, vectors=True):
    base_path = to_path(path, ".nlabel")

    with open(base_path / "meta.json", "r") as f:
        meta = orjson.loads(f.read())

    if meta['type'] != 'document':
        raise RuntimeError(f'expected document, got "{meta["type"]}"')
    if meta['engine'] != 'bahia':
        raise RuntimeError(f'unknown engine "{meta["engine"]}"')
    if meta['version'] != 1:
        raise RuntimeError('unsupported version')

    with open(base_path / 'document.json', 'rb') as f:
        data = orjson.loads(f.read())

    vectors_path = None
    if vectors:
        vectors_path = base_path / "vectors.h5"

    if vectors_path and vectors_path.exists():
        vectors_data = []
        with h5py.File(vectors_path, "r") as f:
            for i in sorted([int(x) for x in f.keys()]):
                group = f[str(i)]
                vectors_data.append(group)

            data['vectors'] = vectors_data
            yield Group(data)
    else:
        yield Group(data)


def save_doc(doc, path, exist_ok=False):
    base_path = to_path(path, ".nlabel")

    if not exist_ok and base_path.exists():
        raise RuntimeError(f"{base_path} exists")

    if not base_path.exists():
        base_path.mkdir()

    doc_guid = make_archive_guid()
    with open(base_path / "meta.json", "wb") as f:
        f.write(orjson.dumps({
            'type': 'document',
            'engine': 'bahia',
            'version': 1,
            'guid': doc_guid
        }))

    json_data, vectors_data = split_data(doc.data)

    with open(base_path / "document.json", "wb") as f:
        f.write(orjson.dumps(json_data))

    if vectors_data:
        with h5py.File(base_path / "vectors.h5", "w") as f:
            for i, x in enumerate(vectors_data):
                group = f.create_group(str(i))
                for k, arr in x.items():
                    group.create_dataset(k, data=arr)
