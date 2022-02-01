import collections
import functools
import json
import logging
import traceback
import hashlib
import uuid

from cached_property import cached_property

from nlabel.io.json.group import split_data
from nlabel.nlp.nlp import NLP as CoreNLP, Text as CoreText
from nlabel.io.carenero.schema import Tagger, Tags, Text, Vector, Vectors, ResultStatus, Result


class TaggerFactory:
    def __init__(self, session):
        self._session = session

    @functools.lru_cache(maxsize=8)
    def from_instance(self, nlp):
        return self.from_data(nlp.description)

    def from_data(self, data):
        return self._from_data_cached(
            json.dumps(data, sort_keys=True))

    @functools.lru_cache(maxsize=8)
    def _from_data_cached(self, description):
        instance = self._session.query(Tagger).filter_by(
            description=description).first()
        if instance:
            return instance
        else:
            instance = Tagger(
                guid=str(uuid.uuid4()).upper(),
                description=description)
            self._session.add(instance)
            self._session.commit()
            return instance


class ExternalKey:
    def __init__(self, value):
        self._value = value
        if isinstance(value, str):
            self._type = 'str'
        elif isinstance(value, (tuple, list, dict)):
            self._type = 'json'
        else:
            raise ValueError(
                f"external key {value} has illegal type {type(value)}")

    @staticmethod
    def from_value(value):
        if value is None:
            return None
        else:
            return ExternalKey(value)

    @property
    def raw(self):
        return self._value

    @property
    def str(self):
        if self._type == 'str':
            return self._value
        else:
            return json.dumps(self._value, sort_keys=True)

    @property
    def type(self):
        return self._type


def text_hash_code(text):
    return hashlib.blake2b(
        text.encode("utf8"), digest_size=32).hexdigest()


class Adder:
    def __init__(self, session, x_tagger, doc):
        text = doc.text
        external_key = doc.external_key
        meta = doc.meta

        if external_key is None:
            raise ValueError("doc needs external_key")

        self._session = session
        self._x_tagger = x_tagger
        self._text = text
        self._meta = meta
        self._external_key = ExternalKey(external_key)

    @property
    def is_duplicate_entry(self):
        return self.x_text is False

    @property
    def x_tagger(self):
        return self._x_tagger

    @cached_property
    def meta_flat(self):
        return json.dumps(self._meta, sort_keys=True) if self._meta else ''

    @cached_property
    def x_text(self):
        batch_text = self._session.query(Text).filter(
            Text.external_key == self._external_key.str,
            Text.external_key_type == self._external_key.type).first()

        if batch_text:
            if batch_text.text != self._text:
                logging.debug(f"TEXT 1: {batch_text.text}")
                logging.debug(f"TEXT 2: {self._text}")

                raise RuntimeError(
                    f"text data for external key '{self._external_key.raw}' does not match")
            if batch_text.meta != self.meta_flat:
                raise RuntimeError(
                    f"meta data for external key '{self._external_key.raw}' does not match")

            if any(r.tagger_id == self._x_tagger.id for r in batch_text.results):
                logging.debug(f"skipping {self._external_key.raw}")
                return False
        else:
            batch_text = Text(
                external_key=self._external_key.str,
                external_key_type=self._external_key.type,
                text=self._text,
                text_hash_code=text_hash_code(self._text),
                meta=self.meta_flat)

        return batch_text

    def make_result(self, doc):
        f = LocalResultFactory(self._x_tagger, self.x_text)
        return f.make_succeeded(doc)


class ResultFactory:
    def _make_succeeded(self, json_data, vectors_data):
        raise NotImplementedError()

    def _check_tagger(self, tagger):
        raise NotImplementedError()

    def make_succeeded(self, doc):
        json_data, vectors_data = split_data(doc.data)

        assert len(json_data['taggers']) == 1
        assert len(vectors_data) == 1
        assert 'vectors' not in json_data
        assert 'error' not in json_data['taggers'][0]

        kept_keys = set(json_data.keys()) - {
            'external_key', 'text', 'meta'}

        json_data = dict(
            (k, json_data[k]) for k in kept_keys)

        assert 'tags' not in json_data
        json_data['tags'] = json_data['taggers'][0]['tags']
        self._check_tagger(json_data['taggers'][0]['tagger'])
        del json_data['taggers']

        return self._make_succeeded(json_data, vectors_data)

    def make_failed(self, err):
        raise NotImplementedError()


def json_to_result(tagger, text, status, json_data):
    tags = json_data.get('tags')

    if tags is not None:
        assert isinstance(tags, dict)

        x_tags = [
            Tags(tag_name=k, data=json.dumps(v))
            for k, v in tags.items()]

        core_json_data = dict((k, v) for k, v in json_data.items() if k != 'tags')
    else:
        x_tags = []
        core_json_data = json_data

    return Result(
        tagger=tagger,
        text=text,
        status=status,
        content=json.dumps(core_json_data),
        tags=x_tags)


class LocalResultFactory(ResultFactory):
    def __init__(self, x_tagger, x_text):
        self._x_tagger = x_tagger
        self._x_text = x_text

    def _check_tagger(self, tagger):
        assert self._x_tagger.description == json.dumps(
            tagger, sort_keys=True)

    def _make_succeeded(self, json_data, vectors_data):
        result = json_to_result(
            tagger=self._x_tagger,
            text=self._x_text,
            status=ResultStatus.succeeded,
            json_data=json_data)

        dtype = '<f4'
        for nlp_vectors_data in vectors_data:
            for k, v in nlp_vectors_data.items():
                if any(x.size == 0 for x in v):
                    raise ValueError(f'vectors contain empty vectors: {v}')
                x_vectors = [Vector(index=i, data=x.astype(dtype).tobytes()) for i, x in enumerate(v)]
                result.vectors.append(Vectors(name=k, dtype=dtype, vectors=x_vectors))

        return result

    def make_failed(self, err):
        return Result(
            text=self._x_text,
            tagger=self._x_tagger,
            status=ResultStatus.failed,
            content=err)


def gen_message(nlp: CoreNLP, text: CoreText):
    doc = None
    message = None

    try:
        doc = nlp(
            text.text,
            meta=text.meta,
            external_key=text.external_key)
    except KeyboardInterrupt:
        raise
    except:
        message = dict(
            text=text,
            doc=None,
            err=json.dumps({
                'traceback': traceback.format_exc()
            }))

    if message is None:
        message = dict(
            text=text, doc=doc.collection, err=None)

    return message


def add_message(session, x_tagger, adder, message):
    if adder.is_duplicate_entry:
        return False

    f = LocalResultFactory(x_tagger, adder.x_text)
    if message.get('doc') is not None:
        result = f.make_succeeded(message['doc'])
    else:
        result = f.make_failed(message['err'])

    with session.no_autoflush:
        assert result is not None
        session.add(result)
        session.commit()

    return True
