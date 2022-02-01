import collections
import h5py
import json
import uuid
import zipfile

from nlabel.io.json.group import split_data


def make_archive(taggers, keyed_docs, path, export_keys=True, compression=None):
    if compression is False:
        json_compression = zipfile.ZIP_STORED
        json_compresslevel = 0
        vectors_compression = None
    elif compression is None:
        json_compression = zipfile.ZIP_DEFLATED
        json_compresslevel = 4
        vectors_compression = 'lzf'
    else:
        json_compression = compression.get('json', {}).get('algorithm', zipfile.ZIP_DEFLATED)
        json_compresslevel = compression.get('json', {}).get('level', 4)
        vectors_compression = compression.get('vectors', 'lzf')

    any_vectors = False

    archive_guid = str(uuid.uuid4()).upper()
    with open(path / "meta.json", "w") as f:
        f.write(json.dumps({
            'type': 'archive',
            'engine': 'bahia',
            'version': 1,
            'guid': archive_guid,
            'taggers': [x.as_meta() for x in taggers]
        }))

    vectors_path = path / "vectors.h5"
    assert not vectors_path.exists()
    try:
        with h5py.File(vectors_path, "w") as vf:
            vf.attrs['archive'] = archive_guid

            with zipfile.ZipFile(
                    path / "documents.zip", "w",
                    compression=json_compression,
                    compresslevel=json_compresslevel) as zf:

                if export_keys:
                    external_keys = collections.defaultdict(list)
                else:
                    external_keys = None

                for i, (key, doc) in enumerate(keyed_docs):
                    if export_keys:
                        external_keys[json.dumps(key, sort_keys=True)].append(i)

                    json_data, vectors_data = split_data(doc.data)

                    if any(x for x in vectors_data):
                        any_vectors = True
                        doc_group = vf.create_group(str(i))
                        for j, nlp_vectors_data in enumerate(vectors_data):
                            if nlp_vectors_data:
                                nlp_group = doc_group.create_group(str(j))
                                for k, v in nlp_vectors_data.items():
                                    nlp_group.create_dataset(
                                        k, data=v,
                                        compression=vectors_compression)

                    zf.writestr(f"{i}.json", json.dumps(json_data))

                if export_keys:
                    zf.writestr(f"keys.json", json.dumps(external_keys))

    finally:
        if not any_vectors:
            vectors_path.unlink(missing_ok=True)
