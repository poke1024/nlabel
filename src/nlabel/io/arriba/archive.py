import contextlib
import orjson
import mmap
import struct
import h5py

from tqdm import tqdm

from ..common import binary_searcher
from ..form import inflected_tag_forms
from ..selector import make_selector
from nlabel.io.arriba.document import Code, DocData, Document
from nlabel.io.arriba.label import factories as label_factories
from nlabel.io.arriba.schema import load_schema


archive_proto = load_schema()


class Iterator:
    def __init__(self, *selectors, archive, mm, inherit_labels, traversal_limit_multiplier, vf):
        self._archive = archive
        self._mm = mm
        self._traversal_limit_multiplier = traversal_limit_multiplier
        self._vf = vf

        selector = make_selector(label_factories, selectors)

        codes = self._archive.codes
        tagger_specs = []
        for tagger in self._archive.taggers:
            tagger_specs.append({
                'tagger': orjson.loads(tagger.spec),
                'tags': dict((x.name, (i, x)) for i, x in [(i, codes[i]) for i in tagger.codes])
            })

        mapped_codes = {}

        def add(nlp_index, form, tags_data):
            code_id, b_code = tags_data
            mapped_codes[code_id] = (form, b_code)

        tag_forms = selector.build(tagger_specs, add)

        self._plural_tag_forms = inflected_tag_forms(tag_forms)

        self._mapped_codes = mapped_codes

    def close(self):
        self._mm = None

    def iter(self, b_doc_refs):
        vf = self._vf

        for doc_i, b_doc_ref in enumerate(b_doc_refs):
            doc_mem = self._mm[b_doc_ref.start:b_doc_ref.end]

            try:
                traversal_limit = len(doc_mem) * self._traversal_limit_multiplier

                with archive_proto.Document.from_bytes(
                        doc_mem,
                        traversal_limit_in_words=traversal_limit) as b_doc:

                    b_tags = b_doc.tags
                    b_codes = binary_searcher([x.code for x in b_tags])

                    tags = {}
                    for code_id, r in self._mapped_codes.items():
                        i = b_codes(code_id)
                        if i is not None:
                            tag_form, b_code = r
                            tags[tag_form.name.external] = Code(
                                code_id,
                                tag_form,
                                b_code,
                                b_tags[i])

                    vectors_group = None if vf is None else vf.get(str(doc_i))

                    doc_data = DocData(b_doc, tags, self._plural_tag_forms, vectors_group)
                    yield Document(doc_data)
            except:
                raise RuntimeError(f"error while reading document {doc_i}")

            finally:
                doc_mem.release()


class Archive:
    def __init__(self, path, b_archive, mm, traversal_limit_multiplier, vectors_file=None):
        self._path = path
        self._archive = b_archive
        self._mm = mm
        self._traversal_limit_multiplier = traversal_limit_multiplier
        self._vectors_file = vectors_file

    def close(self):
        self._mm = None

    def __len__(self):
        return len(self._archive.documents)

    @property
    def path(self):
        return self._path

    def iter(self, *selectors, inherit_labels=True, progress=True):
        it = Iterator(
            *selectors,
            inherit_labels=inherit_labels,
            archive=self._archive,
            mm=self._mm,
            traversal_limit_multiplier=self._traversal_limit_multiplier,
            vf=self._vectors_file)
        try:
            for doc in it.iter(tqdm(self._archive.documents, disable=not progress)):
                yield doc
        finally:
            it.close()


@contextlib.contextmanager
def open_archive(path, traversal_limit_multiplier=1024, mode="r", archive_guid=None):
    if mode != "r":
        raise RuntimeError(f"mode {mode} is not supported")

    with open(path / "archive.bin", "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            archive = None
            mv = memoryview(mm)
            try:
                header_size = struct.unpack("<Q", mv[0:8])[0]
                with archive_proto.Archive.from_bytes(
                        mv[8:8 + header_size],
                        traversal_limit_in_words=len(mv) * traversal_limit_multiplier) as arch:

                    mv2 = mv[8 + header_size:]
                    try:
                        archive_args = [path, arch, mv2, traversal_limit_multiplier]

                        vectors_path = path / "vectors.h5"
                        if vectors_path.exists():
                            with h5py.File(vectors_path, "r") as vf:
                                if vf.attrs['archive'] != archive_guid:
                                    raise RuntimeError("archive GUID mismatch")
                                yield Archive(*archive_args, vectors_file=vf)
                        else:
                            yield Archive(*archive_args)

                    finally:
                        mv2.release()

            finally:
                if archive:
                    archive.close()
                mv.release()
