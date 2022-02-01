import time
import datetime

from typing import List, Union

from nlabel.nlp.tagger.spacy import SpacyTagger
from nlabel.nlp.tagger.stanza import StanzaTagger
from nlabel.nlp.tagger.flair import FlairTagger
from nlabel.nlp.tagger.pavlov import PavlovTagger
from nlabel.nlp.core import Tagger, Text
from nlabel.io.json.group import Group
from nlabel.io.selector import One


class NLP:
    _default_taggers = (
        SpacyTagger,
        StanzaTagger,
        FlairTagger,
        PavlovTagger
    )

    @staticmethod
    def _pick_tagger(nlp, vectors=False, meta=None, require_gpu=False, **kwargs):
        good = None
        for c in NLP._default_taggers:
            if c.is_compatible_nlp(nlp):
                good = c(
                    nlp,
                    vectors=vectors,
                    meta=meta,
                    require_gpu=require_gpu,
                    **kwargs)
                break
        if not good:
            raise RuntimeError(
                f"nlp {nlp} is currently not supported")
        return good

    def __init__(self, nlp, vectors=False, meta=None, require_gpu=False, **kwargs):
        if isinstance(nlp, Tagger):
            if vectors or meta or require_gpu or kwargs:
                raise ValueError("got unused parameters")
            self._tagger = nlp
        else:
            self._tagger = self._pick_tagger(
                nlp, vectors=vectors, meta=meta, require_gpu=require_gpu, **kwargs)

    @staticmethod
    def spacy(nlp, **kwargs):
        return NLP(SpacyTagger(nlp, **kwargs))

    @staticmethod
    def flair(*taggers, from_spacy=None, **kwargs):
        import flair.models
        tagger = flair.models.MultiTagger.load(list(taggers))
        return NLP(FlairTagger(tagger, from_spacy=from_spacy, **kwargs))

    @staticmethod
    def stanza(nlp, **kwargs):
        return NLP(StanzaTagger(nlp, **kwargs))

    @staticmethod
    def deeppavlov(nlp, **kwargs):
        return NLP(PavlovTagger(nlp, **kwargs))

    @property
    def description(self):
        return self._tagger.description

    def _process(self, text):
        t0 = time.time()
        builder = self._tagger.process(text)
        t1 = time.time()

        raw_data = builder.data
        raw_data['stat'] = {
            'elapsed': t1 - t0,
            'created_at': datetime.datetime.utcnow().isoformat()
        }

        return raw_data, builder.vectors_data

    def _process_n(self, texts, batch_size=1):
        return [
            (x.data, x.vectors_data)
            for x in self._tagger.process_n(texts, batch_size=batch_size)]

    def _make_doc(self, built_data, item_data):
        raw_data, raw_vectors_data = built_data

        data = {
            'text': item_data['text'],
            'taggers': [raw_data],
            'vectors': [raw_vectors_data]
        }

        meta = item_data.get('meta')
        if meta is not None:
            data['meta'] = meta

        external_key = item_data.get('external_key')
        if external_key is not None:
            data['external_key'] = external_key

        return Group(data).view(One())

    def __call__(self, text: str, meta: dict = None, external_key: Union[str, dict] = None):
        return self._make_doc(
            self._process(text), {
                'text': text,
                'meta': meta,
                'external_key': external_key
            })

    def pipe(self, texts: List[Text], batch_size: int = 1):
        texts = list(texts)
        data = self._process_n(
            [x.text for x in texts],
            batch_size=batch_size)
        for built_data, text in zip(data, texts):
            yield self._make_doc(
                built_data, text._asdict())
