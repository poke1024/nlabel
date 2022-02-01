import functools

import numpy as np
import codecs
import collections
import dbm
import itertools
import hashlib
import orjson
import json
import uuid
import h5py
import struct
import sqlalchemy

from nlabel.io.json.group import split_data
from nlabel.io.common import AbstractSpanFactory
from nlabel.io.arriba.schema import load_schema
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from pathlib import Path


archive_proto = load_schema()

Base = declarative_base()


class IndexEntry(Base):
    __tablename__ = 'index'

    id = Column(Integer, primary_key=True)
    key = Column(String, nullable=False, index=True)
    doc = Column(Integer, nullable=False)


class TempSpan(collections.namedtuple('TempSpan', ['start', 'end'])):
    @property
    def length(self):
        return self.end - self.start


class SpanFactory(AbstractSpanFactory):
    def __init__(self, text):
        super().__init__()
        w = np.array([len(x) for x in codecs.iterencode(text, "utf8")], dtype=np.uint32)
        self._offsets = np.cumsum(np.concatenate([[0], w]))

    def _make_span(self, start, end):
        return TempSpan(
            start=int(self._offsets[start]),
            end=int(self._offsets[end]))


class Lexicon:
    def __init__(self):
        self._inverse = {}
        self._sequence = []

    def _make(self, index, value):
        return index, value

    def add(self, value):
        x = self._inverse.get(value)
        if x is None:
            x, y = self._make(len(self._sequence), value)
            self._inverse[value] = x
            self._sequence.append(y)
        return x

    @property
    def sequence(self):
        return self._sequence


class ObjectLexicon(Lexicon):
    def __init__(self, klass):
        super().__init__()
        self._klass = klass

    def _make(self, index, value):
        x = self._klass(index, value)
        return x, x


class TempLabel(collections.namedtuple('TempLabel', ['value', 'score'])):
    def make_label(self):
        return archive_proto.ScoredLabel.new_message(
            value=self.value,
            score=self.score)


class BuildTagger:
    def __init__(self, id_, key):
        nlp_id, name = key
        self._id = id_
        self._nlp_id = nlp_id
        self._name = name
        self._values = Lexicon()

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def values(self):
        return self._values.sequence

    @property
    def nlp_id(self):
        return self._nlp_id

    def register_value(self, value):
        return self._values.add(value)

    def make_label(self, label):
        return TempLabel(
            value=self.register_value(label.get("value")),
            score=label.get('score'))


TempTag = collections.namedtuple('TempTag', ['span', 'labels', 'parent'])


def make_temp_tag(data, spans_id):
    span = data.get('span')
    return TempTag(
        spans_id[id(span)] if span else -1,
        data['labels'],
        data['parent'] or -1)


class RefsFactory:
    def __init__(self):
        pass

    @functools.lru_cache
    def _make(self, seq):
        if len(seq) > 0:
            seq_max = np.max(seq)
        else:
            seq_max = 0
        r = archive_proto.SharedListInt()
        if seq_max <= np.iinfo(np.int8).max:
            r.values.int8 = seq
        elif seq_max <= np.iinfo(np.int16).max:
            r.values.int16 = seq
        else:
            r.values.int32 = seq
        return r

    def __call__(self, seq):
        return self._make(tuple(seq))


def write_packed(target, values, allow_uint64=False):
    values = np.array(values)
    values_max = np.max(values)
    if values_max <= 0xff:
        target.uint8 = values.tolist()
    elif values_max <= 0xffff:
        target.uint16 = values.tolist()
    elif values_max <= 0xffffffff:
        target.uint32 = values.tolist()
    else:
        assert allow_uint64
        target.uint64 = values.tolist()


def write_labels(target, values, scores, default_score=np.nan):
    if len(values) == 0:
        target.values.none = None
        target.scores.none = None
    else:
        write_packed(target.values, values)
        if all(x is None for x in scores):
            target.scores.none = None
        else:
            target.scores.float32 = [
                (x or default_score) for x in scores]


def make_sorted_tags(tags, spans_id):
    temp_tags = [make_temp_tag(x, spans_id) for x in tags]
    return sorted(temp_tags, key=lambda x: x.span)


class TempTagger:
    def __init__(self, tagger_id, vectors_data):
        self._tagger_id = tagger_id

        self._tags = []
        self._sorted_tags = None

        self._vectors_data = vectors_data
        self._span_vi = None if vectors_data is None else {}

    def add_tag(self, i, labels, parent, span):
        assert self._sorted_tags is None
        data = {
            'labels': labels,
            'parent': parent
        }
        if span is not None:
            data['span'] = span
            if self._span_vi is not None:
                self._span_vi[id(span)] = i
        self._tags.append(data)

    @property
    def tagger_id(self):
        return self._tagger_id

    def sort_tags(self, spans_id):
        assert self._sorted_tags is None
        self._sorted_tags = make_sorted_tags(self._tags, spans_id)

    @property
    def sorted_tags(self):
        assert self._sorted_tags is not None
        return self._sorted_tags

    def make_tags(self, make_refs):
        sorted_tags = self.sorted_tags

        r = archive_proto.CodeData(
            code=self.tagger_id,
            spans=make_refs([x.span for x in sorted_tags]))

        labels = [x.labels for x in sorted_tags]
        labels_len = np.array([len(x) for x in labels])

        if len(labels) == 0:
            r.labels.groups.none = None
            r.labels.values.none = None
            r.labels.scores.none = None
        elif np.all(labels_len == 1):
            r.labels.groups.none = None
            write_labels(
                r.labels,
                [x[0].value for x in labels],
                [x[0].score for x in labels])
        else:
            write_packed(
                r.labels.groups,
                np.cumsum(np.concatenate([[0], labels_len])))
            write_labels(
                r.labels,
                list(itertools.chain(*[[y.value for y in x] for x in labels])),
                list(itertools.chain(*[[y.score for y in x] for x in labels])))

        parents = [x.parent for x in sorted_tags]

        if all(x < 0 for x in parents):
            r.parents.none = None
        else:
            r.parents.indices = make_refs(parents)

        return r

    def save_vectors(self, reordered_span_ids, vf):
        if self._vectors_data is None:
            return

        span_id_to_vector_id = dict(
            (reordered_span_ids[span_py_id], i)
            for span_py_id, i in self._span_vi.items())

        vectors = []
        for x in self.sorted_tags:
            i = span_id_to_vector_id[x.span]
            vectors.append(self._vectors_data[i])

        vf.create_dataset(
            str(self._tagger_id), data=np.array(vectors))


def make_sorted_taggers(taggers, reordered_span_ids):
    for tagger in taggers:
        tagger.sort_tags(reordered_span_ids)

    refs_factory = RefsFactory()
    return sorted([
        t.make_tags(refs_factory) for t in taggers],
        key=lambda x: x.code)


class Archive:
    def __init__(self, temp_db, index_session, vectors_file):
        self._nlps = Lexicon()
        self._taggers = ObjectLexicon(BuildTagger)
        self._docs = []
        self._temp_db = temp_db
        self._index_session = index_session
        self._vectors_file = vectors_file
        self._data_offset = 0

    def make_nlps(self):
        res = []
        for nlp_id, nlp_data in enumerate(self._nlps.sequence):
            nlp_tagger_ids = []
            for tagger_id, tagger in enumerate(self._taggers.sequence):
                if tagger.nlp_id == nlp_id:
                    nlp_tagger_ids.append(tagger_id)
            res.append(archive_proto.Tagger.new_message(
                spec=nlp_data,
                codes=nlp_tagger_ids))
        return res

    def make_taggers(self):
        res = []
        for tagger_id, tagger in enumerate(self._taggers.sequence):
            res.append(archive_proto.Code.new_message(
                tagger=tagger.nlp_id,
                name=tagger.name,
                values=tagger.values))
        return res

    def register_nlp(self, nlp):
        return self._nlps.add(orjson.dumps(nlp))

    def register_tagger(self, nlp_id, tag_name):
        return self._taggers.add((nlp_id, tag_name))

    def _add_index(self, utf8_text, external_key):
        if external_key:
            self._index_session.add(IndexEntry(
                key=external_key, doc=len(self._docs)))

        self._index_session.add(IndexEntry(
            key=hashlib.blake2b(utf8_text, digest_size=4).hexdigest(),
            doc=len(self._docs)))

    def add_doc(self, doc, external_key=None):
        data, vectors_data = split_data(doc.data)

        span_factory = SpanFactory(data["text"])
        temp_taggers = {}

        for nlp_data, nlp_vectors_data in zip(data["taggers"], vectors_data):
            nlp_id = self.register_nlp(nlp_data["tagger"])
            for tag_name, tag_data in nlp_data["tags"].items():
                tagger = self.register_tagger(nlp_id, tag_name)

                assert tagger.id not in temp_taggers
                temp_tagger = TempTagger(
                    tagger.id, nlp_vectors_data.get(tag_name))
                temp_taggers[tagger.id] = temp_tagger

                for tag_i, tag in enumerate(tag_data):
                    temp_tagger.add_tag(
                        i=tag_i,
                        labels=[tagger.make_label(x) for x in tag.get("labels", [])],
                        parent=tag.get("parent"),
                        span=span_factory.from_json(tag))

        sorted_spans = span_factory.sorted_spans
        reordered_span_ids = dict((id(span), i) for i, span in enumerate(sorted_spans))

        tagger_tags = make_sorted_taggers(
            list(temp_taggers.values()), reordered_span_ids)

        if self._vectors_file is not None:
            group = self._vectors_file.create_group(str(len(self._docs)))
            for tagger in temp_taggers.values():
                tagger.save_vectors(reordered_span_ids, group)

        utf8_text = data["text"].encode("utf8")
        self._add_index(utf8_text, external_key)

        p_doc = archive_proto.Document.new_message(
            text=utf8_text,
            meta=orjson.dumps(data.get("meta", "")),
            tags=tagger_tags)

        write_packed(p_doc.starts, [s.start for s in sorted_spans], allow_uint64=True)
        write_packed(p_doc.lens, [s.length for s in sorted_spans], allow_uint64=True)

        doc_data = p_doc.to_bytes()

        self._temp_db[str(len(self._docs))] = doc_data

        self._docs.append(archive_proto.DocumentRef.new_message(
            start=self._data_offset,
            end=self._data_offset + len(doc_data)
        ))
        self._data_offset += len(doc_data)

    def commit(self):
        self._index_session.commit()

    def save_archive(self, f):
        self.commit()

        b_archive = archive_proto.Archive.new_message(
            version=1,
            taggers=self.make_nlps(),
            codes=self.make_taggers(),
            documents=self._docs)
        b_archive_data = b_archive.to_bytes()

        header_size = len(b_archive_data)
        f.write(struct.pack("<Q", header_size))
        f.write(b_archive_data)

        for i in range(len(self._docs)):
            doc_data = self._temp_db[str(i)]
            f.write(doc_data)
            assert len(doc_data) == self._docs[i].end - self._docs[i].start


def make_session(path):
    engine = sqlalchemy.create_engine(f"sqlite:///{path}")

    # see https://docs.sqlalchemy.org/en/14/dialects/sqlite.html#pysqlite-serializable

    @sqlalchemy.event.listens_for(engine, "connect")
    def do_connect(dbapi_connection, connection_record):
        # disable pysqlite's emitting of the BEGIN statement entirely.
        # also stops it from emitting COMMIT before any DDL.
        dbapi_connection.isolation_level = None

    @sqlalchemy.event.listens_for(engine, "begin")
    def do_begin(conn):
        # emit our own BEGIN
        conn.exec_driver_sql("BEGIN")

    Session = sqlalchemy.orm.sessionmaker()
    Session.configure(bind=engine)
    session = Session()

    Base.metadata.create_all(engine)
    return session


def make_archive(keyed_docs, path: Path, commit_freq: int = 500):
    dbs = [
        'index'
    ]

    db_paths = {}
    db_sessions = {}

    for db in dbs:
        db_path = path / f"{db}.sqlite"
        if db_path.exists():
            raise RuntimeError(f"{db_path} already exists")
        db_paths[db] = db_path

    vectors_path = path / "vectors.h5"
    with h5py.File(vectors_path, "w") as vf:

        try:
            for db in dbs:
                db_sessions[db] = make_session(db_paths[db])

            with dbm.open(str(path / 'temp'), 'n') as temp_db:

                archive = Archive(
                    temp_db,
                    db_sessions['index'],
                    vf)
                for i, (external_key, doc) in enumerate(keyed_docs):
                    archive.add_doc(
                        doc,
                        external_key=json.dumps(external_key, sort_keys=True))
                    if i % commit_freq == 0:
                        archive.commit()
                with open(path / "archive.bin", 'wb') as f:
                    archive.save_archive(f)

            archive_guid = str(uuid.uuid4()).upper()
            with open(path / "meta.json", "w") as f:
                f.write(json.dumps({
                    'type': 'archive',
                    'engine': 'arriba',
                    'version': 1,
                    'guid': archive_guid
                }))

            vf.attrs['archive'] = archive_guid

        except:
            for session in db_sessions.values():
                session.close()
            db_sessions = {}

            for db_path in db_paths.values():
                db_path.unlink(missing_ok=True)

            raise

        finally:
            for p in list(path.iterdir()):
                if p.stem == 'temp':
                    p.unlink()

            for session in db_sessions.values():
                session.close()
            db_sessions = {}
