import functools
import logging
import traceback
import hashlib
import orjson

from cached_property import cached_property

from nlabel.io.json.group import split_data
from nlabel.nlp.nlp import NLP as CoreNLP, Text as CoreText
from nlabel.io.carenero.schema import Tagger, Tag, TagInstances, Text, Vector, Vectors, ResultStatus, Result
from ..guid import tagger_guid, text_guid


class TaggerFactory:
    def __init__(self, session):
        self._session = session

    @functools.lru_cache(maxsize=8)
    def from_instance(self, nlp):
        return self.from_data(nlp.signature)

    def from_data(self, data):
        return self._from_data_cached(
            orjson.dumps(data, option=orjson.OPT_SORT_KEYS).decode("utf8"))

    @functools.lru_cache(maxsize=8)
    def _from_data_cached(self, signature):
        instance = self._session.query(Tagger).filter_by(
            signature=signature).first()
        if instance:
            return instance
        else:
            instance = Tagger(
                guid=tagger_guid(),
                signature=signature)
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
            return orjson.dumps(
                self._value,
                option=orjson.OPT_SORT_KEYS).decode("utf8")

    @property
    def type(self):
        return self._type


def _text_diff(x_text, doc):
    if x_text.text != doc.text:
        return 'text'
    elif x_text.meta != doc.meta_json:
        return 'meta'
    else:
        return None


def _check_text_diff(x_text, doc, display_key):
    diff = _text_diff(x_text, doc)
    if diff is not None:
        raise RuntimeError(
            f"new entry with {display_key}'"
            f"does not match existing db entry in terms of {diff}")


class Entry:
    @cached_property
    def guid(self):
        return text_guid()

    def find(self, session):
        raise NotImplementedError()

    @property
    def display_key(self):
        raise NotImplementedError()

    @property
    def external_key(self):
        raise NotImplementedError()


class KeyedEntry(Entry):
    def __init__(self, doc):
        self._doc = doc
        self._external_key = ExternalKey(doc.external_key)

    def find(self, session):
        x_text = session.query(Text).filter(
            Text.external_key == self._external_key.str,
            Text.external_key_type == self._external_key.type).first()

        if x_text:
            _check_text_diff(x_text, self._doc, self.display_key)
            return x_text
        else:
            return None

    @property
    def display_key(self):
        return f"external key '{self._external_key.raw}'"

    @property
    def external_key(self):
        return self._external_key


class UnkeyedEntry(Entry):
    def __init__(self, doc):
        self._doc = doc

    def find(self, session):
        for x_cand in session.query(Text).filter(Text.text_hash_code == self._doc.text_hash_code).yield_per(10):
            if x_cand.text == self._doc.text:
                _check_text_diff(x_cand, self._doc, self.display_key)
                return x_cand

        return None

    @property
    def display_key(self):
        return f"text '{self._doc.text[:20]}...'"

    @property
    def external_key(self):
        return ExternalKey(self.guid)


def make_entry(doc):
    if doc.external_key is not None:
        return KeyedEntry(doc)
    else:
        return UnkeyedEntry(doc)


class Adder:
    def __init__(self, session, x_tagger, doc):
        self._session = session
        self._x_tagger = x_tagger
        self._doc = doc
        self._text = doc.text
        self._meta = doc.meta
        self._entry = make_entry(doc)

    @property
    def is_duplicate_text(self):
        return self.x_text is False

    @property
    def x_tagger(self):
        return self._x_tagger

    @cached_property
    def x_text(self):
        x_text = self._entry.find(self._session)

        if x_text:
            if any(r.tagger_id == self._x_tagger.id for r in x_text.results):
                logging.debug(f"skipping {self._entry.display_key}")
                return False
        else:
            external_key = self._entry.external_key

            x_text = Text(
                guid=self._entry.guid,
                external_key=external_key.str,
                external_key_type=external_key.type,
                text=self._text,
                text_hash_code=self._doc.text_hash_code,
                meta=self._doc.meta_json)

        return x_text

    def make_result(self, doc):
        f = LocalResultFactory(self._x_tagger, self.x_text)
        return f.make_succeeded(doc)


class ResultFactory:
    def _make_succeeded(self, json_data, vectors_data):
        raise NotImplementedError()

    def _check_signature(self, signature):
        raise NotImplementedError()

    def make_succeeded(self, doc):
        json_data, vectors_data = split_data(doc.data)

        assert len(json_data['taggers']) == 1
        assert len(vectors_data) == 1
        assert 'vectors' not in json_data
        assert 'error' not in json_data['taggers'][0]

        kept_keys = set(json_data.keys()) - {
            'external_key', 'root', 'meta'}

        json_data = dict(
            (k, json_data[k]) for k in kept_keys)

        assert 'tags' not in json_data
        json_data['tags'] = json_data['taggers'][0]['tags']
        self._check_signature(json_data['taggers'][0]['tagger'])
        del json_data['taggers']

        return self._make_succeeded(json_data, vectors_data)

    def make_failed(self, err):
        raise NotImplementedError()


def _find_or_create_tag(tagger, name):
    for tag in tagger.tags:
        if tag.name == name:
            return tag
    tag = Tag(tagger=tagger, name=name)
    tagger.tags.append(tag)
    return tag


def json_to_result(tagger, text, status, json_data):
    tags = json_data.get('tags')

    if tags is not None:
        assert isinstance(tags, dict)

        x_tag_i = [
            TagInstances(
                tag=_find_or_create_tag(tagger, k),
                data=orjson.dumps(v).decode("utf8"))
            for k, v in tags.items()]

        core_json_data = dict((k, v) for k, v in json_data.items() if k != 'tags')
    else:
        x_tag_i = []
        core_json_data = json_data

    return Result(
        tagger=tagger,
        text=text,
        status=status,
        data=orjson.dumps(core_json_data).decode("utf8"),
        tag_instances=x_tag_i)


class LocalResultFactory(ResultFactory):
    def __init__(self, x_tagger, x_text):
        self._x_tagger = x_tagger
        self._x_text = x_text
        self._signature = x_tagger.signature.encode("utf8")

    def _check_signature(self, signature):
        assert self._signature == orjson.dumps(
            signature, option=orjson.OPT_SORT_KEYS)

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
            data=err)


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
            err=orjson.dumps({
                'traceback': traceback.format_exc()
            }).decode("utf8"))

    if message is None:
        message = dict(
            text=text, doc=doc.collection, err=None)

    return message


def add_message(session, x_tagger, adder, message):
    if adder.is_duplicate_text:
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
