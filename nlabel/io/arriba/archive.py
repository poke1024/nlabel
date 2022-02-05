import contextlib
import orjson
import mmap
import struct
import h5py

from tqdm import tqdm
from cached_property import cached_property

from ..common import binary_searcher
from ..form import inflected_tag_forms
from ..selector import make_selector, auto_selectors
from nlabel.io.json.group import TaggerPrivate as JsonTagger, TaggerList as JsonTaggerList
from nlabel.io.arriba.document import Code, DocData, Document
from nlabel.io.arriba.label import factories as label_factories
from nlabel.io.arriba.schema import load_schema

archive_proto = load_schema()


@contextlib.contextmanager
def _capnproto_obj(mvs, obj):
    # this is a workaround to work with the current version of pycapnp. hopefully,
    # https://github.com/capnproto/pycapnp/pull/281 will be available in the future.

    if isinstance(obj, contextlib._GeneratorContextManager):
        try:
            with obj as x:
                yield x
        finally:
            for mv in mvs:
                mv.release()
    else:
        # will leak memory, since mv is not closed properly.
        yield obj


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
                'guid': tagger.guid,
                'tagger': orjson.loads(tagger.signature),
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

                with _capnproto_obj([doc_mem], archive_proto.Document.from_bytes(
                        doc_mem,
                        traversal_limit_in_words=traversal_limit)) as b_doc:

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


class Archive:
    def __init__(self, path, taggers, b_archive, mm, traversal_limit_multiplier, vectors_file=None):
        self._path = path
        self._taggers = taggers
        self._archive = b_archive
        self._mm = mm
        self._traversal_limit_multiplier = traversal_limit_multiplier
        self._vectors_file = vectors_file

    @property
    def engine(self):
        return "arriba"

    def close(self):
        self._mm = None

    def __len__(self):
        return len(self._archive.documents)

    @property
    def path(self):
        return self._path

    @cached_property
    def taggers(self):
        return JsonTaggerList([
            JsonTagger.from_meta(x) for x in self._taggers])

    def iter(self, *selectors, inherit_labels=True, progress=True):
        selectors = auto_selectors(selectors, self.taggers)

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
def open_archive(info, traversal_limit_multiplier=1024):
    path = info.base_path

    if info.mode != "r":
        raise RuntimeError(f"mode {info.mode} is not supported")

    with open(path / "archive.bin", "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            mv = memoryview(mm)

            header_size = struct.unpack("<Q", mv[0:8])[0]
            mv1 = mv[8:8 + header_size]
            mv2 = mv[8 + header_size:]

            with _capnproto_obj([mv1, mv2, mv], archive_proto.Archive.from_bytes(
                    mv1,
                    traversal_limit_in_words=len(mv) * traversal_limit_multiplier)) as arch:

                archive_args = [path, info.taggers, arch, mv2, traversal_limit_multiplier]

                vectors_path = path / "vectors.h5"
                if vectors_path.exists():
                    with h5py.File(vectors_path, "r") as vf:
                        if vf.attrs['archive'] != info.guid:
                            raise RuntimeError("archive GUID mismatch")
                        yield Archive(*archive_args, vectors_file=vf)
                else:
                    yield Archive(*archive_args)
