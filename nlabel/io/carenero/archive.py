import json
import traceback

import orjson
import contextlib
import concurrent.futures
import numpy as np
import logging
import queue

from tqdm import tqdm
from typing import List, Union

from nlabel.io.common import make_writer
from nlabel.io.json.loader import Loader
from nlabel.io.json.group import Group, Tagger as JsonTagger, TaggerList as JsonTaggerList
from nlabel.io.json.archive import Archive as AbstractArchive
from nlabel.nlp.core import Text as CoreText
from nlabel.nlp.nlp import NLP as CoreNLP
from nlabel.io.carenero.schema import create_session_factory, \
    Text, ResultStatus, Result, Tagger
from nlabel.io.carenero.common import TaggerFactory, Adder, \
    gen_message, add_message
from nlabel.io.json import Document


def _result_to_doc(result, vectors=True, migrate=None):
    text = result.text

    json_data = orjson.loads(result.data)

    if migrate is not None:
        json_data = migrate(json_data)

    json_data['text'] = text.text
    if text.meta:
        json_data['meta'] = orjson.loads(text.meta)
    json_data['guid'] = text.guid

    if 'tags' in json_data:
        assert not result.tags
    else:
        tags_data = {}
        for tag_i in result.tag_instances:
            tags_data[tag_i.tag.name] = orjson.loads(tag_i.data)
        json_data['tags'] = tags_data

    assert 'taggers' not in json_data
    json_data['taggers'] = [{
        'guid': result.tagger.guid,
        'tagger': result.tagger.signature_as_dict,
        'tags': json_data['tags']
    }]
    del json_data['tags']

    json_data['external_key'] = text.decoded_external_key

    if vectors:
        vectors_data = {}
        for vectors in result.vectors:
            arr = []
            dtype = vectors.dtype
            for i, vector in enumerate(vectors.vectors):
                assert vector.index == i
                arr.append(np.frombuffer(vector.data, dtype=dtype))
            vectors_data[vectors.name] = np.array(arr)
        json_data['vectors'] = [vectors_data]

    return Group(json_data)


class Exporter:
    def __init__(
            self, archive, migrate=None, join_nlps=True,
            allow_failed=False, allow_empty=False,
            errors_only=False, export_vectors=True,
            force_complete=True):

        if force_complete:
            if not archive.is_complete():
                raise RuntimeError("result data in this archive is incomplete")

        self._migrate = migrate
        self._join_nlps = join_nlps
        self._allow_failed = allow_failed
        self._allow_empty = allow_empty
        self._errors_only = errors_only
        self._export_vectors = export_vectors

    def export(self, text):
        docs = []
        has_err = False

        for result in text.results:
            if result.status == ResultStatus.succeeded:
                docs.append(_result_to_doc(result, vectors=self._export_vectors))
            else:
                if self._allow_failed:
                    err_data = {
                        'text': result.text.text,
                        'taggers': [{
                            'tagger': json.loads(result.tagger.signature),
                            'error': json.loads(result.data)
                        }]
                    }
                    docs.append(Group(err_data))
                    has_err = True
                else:
                    raise RuntimeError(
                        f"encountered failed result on '{text.external_key}': {json.loads(result.data)}")

        if self._errors_only and not has_err:
            return None

        if not docs:
            if self._allow_empty:
                return None
            else:
                raise RuntimeError(f"no data found for '{text.external_key}'")

        if self._join_nlps:
            try:
                yield Group.join(docs)
            except RuntimeError:
                raise RuntimeError(
                    f"cannot join documents with external key '{text.external_key}'")
        else:
            for doc in docs:
                yield doc


class Archive(AbstractArchive):
    def __init__(self, path, session, new_session, mode='r', migrate=None):
        self._path = path
        self._mode = mode
        self._migrate = migrate

        self._session = session
        self._tagger_factory = TaggerFactory(session)

        self._new_session = new_session

    def _assert_write_mode(self):
        if self._mode not in ('w', 'w+', 'r+'):
            raise RuntimeError(f"mode = {self._mode}, not a write mode")

    def _batch_add(self, nlp: CoreNLP, items: List[CoreText], ignore_duplicates=True):
        x_tagger = self._tagger_factory.from_instance(nlp)
        with self._session.no_autoflush:
            for item in items:
                adder = Adder(self._session, x_tagger, item)
                if ignore_duplicates and adder.is_duplicate_text:
                    continue
                message = gen_message(nlp, item)
                if message is None:
                    continue
                add_message(self._session, x_tagger, adder, message)
                self._session.commit()

    def batch_add(self, nlp: CoreNLP, items: List[CoreText], ignore_duplicates=True):
        # * prevents actually computing the nlp when doc is already in archive
        # * records errors during nlp processing into archive

        self._assert_write_mode()
        self._batch_add(nlp, items, ignore_duplicates=ignore_duplicates)

    def add(self, item: Union[Group, Document], ignore_duplicates=True):
        self._assert_write_mode()

        if isinstance(item, Document):
            doc = item.collection
        else:
            doc = item

        any_new = False

        with self._session.no_autoflush:
            for split_doc in doc.split():
                x_tagger = self._tagger_factory.from_data(
                    split_doc.nlps[0]['tagger'])
                adder = Adder(self._session, x_tagger, split_doc)
                if ignore_duplicates and adder.is_duplicate_text:
                    continue
                any_new = True
                self._session.add(adder.make_result(split_doc))
            self._session.commit()

        return any_new

    def del_errors(self):
        if self._mode != 'w':
            raise RuntimeError("not in write mode")

        self._session.query(Result).filter(
            Result.status == ResultStatus.failed).delete()
        self._session.commit()

    def is_complete(self):
        tagger_ids = set()
        for row in self._session.query(Tagger).with_entities(Tagger.id).all():
            tagger_ids.add(row[0])

        counts = {}
        for tagger_id in sorted(tagger_ids):
            counts[tagger_id] = self._session.query(Result).filter(
                Result.tagger_id == tagger_id).count()

        if not counts:
            return True

        expected_n = list(counts.values())[0]
        if not all(x == expected_n for x in counts.values()):
            for tagger_id, n in sorted(counts.items()):
                logging.info(f"tagger {tagger_id}: {n} results")
            return False

        return True

    def _groups(self, progress=True, **kwargs):
        exporter = Exporter(self, migrate=self._migrate, **kwargs)

        n = self._session.query(Text).count()
        query = self._session.query(Text).yield_per(100)

        for text in tqdm(query, total=n, disable=not progress):
            for group in exporter.export(text):
                yield text.decoded_external_key, group

    def _fast_iter(self, *selectors, vectors=True):
        loader = Loader(*selectors)

        query = self._session.query(Result).filter(
            Result.status == ResultStatus.succeeded).yield_per(100)

        for result in query:
            doc = _result_to_doc(
                result,
                vectors=vectors,
                migrate=self._migrate)
            yield loader(doc)

    @property
    def taggers(self):
        taggers = []
        for tagger in self._session.query(Tagger).yield_per(100):
            taggers.append(JsonTagger({
                'guid': tagger.guid,
                'tagger': orjson.loads(tagger.signature),
                'tags': dict((x.name, None) for x in tagger.tags)
            }))
        return JsonTaggerList(taggers)

    def iter(self, *selectors, progress=True):
        loader = Loader(*selectors, taggers=self.taggers)

        for _, group in self._groups(progress=progress):
            yield loader(group)

    def save(self, path, engine, options=None, exist_ok=False, progress=True):
        group_q = queue.Queue(maxsize=16)

        def groups_from_queue():
            while True:
                message = group_q.get()

                if message == "STOP":
                    break

                yield message

        writer = make_writer(path, engine, exist_ok=exist_ok)
        options = writer.set_options(options)

        def save(taggers):
            try:
                writer.write(groups_from_queue(), taggers)
            except:
                traceback.print_exc()
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(save, list(self.taggers))

            for k, group in self._groups(progress=progress, **options):
                group_q.put((k, group))

            group_q.put("STOP")


@contextlib.contextmanager
def open_archive(info, migrate=None, echo_sql=False):
    path = info.base_path
    mode = info.mode

    session_factory = create_session_factory(path, echo=echo_sql)

    session = session_factory()
    try:
        yield Archive(path, session, session_factory, mode=mode, migrate=migrate)
    finally:
        session.close()
