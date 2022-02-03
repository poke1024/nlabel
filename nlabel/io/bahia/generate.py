import collections
import h5py
import orjson
import zipfile

from nlabel.io.json.group import split_data
from nlabel.io.guid import archive_guid as make_archive_guid
from nlabel.io.common import AbstractWriter


class BahiaWriter(AbstractWriter):
    def __init__(self, path, exist_ok=False):
        super().__init__(path, exist_ok=exist_ok)

        self.compression = None
        self.export_keys = True

    def set_options(self, options):
        options = options or {}
        self.compression = options.get('compression')
        self.export_keys = options.get('export_keys', True)
        return dict((k, v) for k, v in options.items() if k not in ('compression', 'export_keys'))

    @property
    def _compression_dict(self):
        if self.compression is False:
            return {
                'zip': {
                    'compression': zipfile.ZIP_STORED,
                    'compresslevel': 0
                },
                'h5': {
                    'compression': None
                }
            }
        elif self.compression is None:
            return {
                'zip': {
                    'compression': zipfile.ZIP_DEFLATED,
                    'compresslevel': 4
                },
                'h5': {
                    'compression': 'lzf'
                }
            }
        else:
            return {
                'zip': {
                    'compression': self.compression.get('zip', {}).get('algorithm', zipfile.ZIP_DEFLATED),
                    'compresslevel': self.compression.get('zip', {}).get('level', 4)
                },
                'h5': {
                    'compression': self.compression.get('h5', 'lzf')
                }
            }

    def _write(self, path, groups, taggers):
        any_vectors = False

        archive_guid = make_archive_guid()
        with open(path / "meta.json", "wb") as f:
            f.write(orjson.dumps({
                'type': 'archive',
                'engine': 'bahia',
                'version': 1,
                'guid': archive_guid,
                'taggers': [x._.as_meta() for x in taggers]
            }))

        vectors_path = path / "vectors.h5"
        assert not vectors_path.exists()
        try:
            with h5py.File(vectors_path, "w") as vf:
                vf.attrs['archive'] = archive_guid

                compression = self._compression_dict
                with zipfile.ZipFile(
                        path / "documents.zip", "w", **compression['zip']) as zf:

                    if self.export_keys:
                        external_keys = collections.defaultdict(list)
                    else:
                        external_keys = None

                    for i, (key, doc) in enumerate(groups):
                        if self.export_keys:
                            external_keys[orjson.dumps(
                                key,
                                option=orjson.OPT_SORT_KEYS).decode("utf8")].append(i)

                        json_data, vectors_data = split_data(doc.data)

                        if any(x for x in vectors_data):
                            any_vectors = True
                            doc_group = vf.create_group(str(i))
                            for j, nlp_vectors_data in enumerate(vectors_data):
                                if nlp_vectors_data:
                                    nlp_group = doc_group.create_group(str(j))
                                    for k, v in nlp_vectors_data.items():
                                        nlp_group.create_dataset(
                                            k, data=v, **compression['h5'])

                        zf.writestr(f"{i}.json", orjson.dumps(json_data))

                    if self.export_keys:
                        zf.writestr(f"keys.json", orjson.dumps(external_keys))

        finally:
            if not any_vectors:
                vectors_path.unlink(missing_ok=True)
