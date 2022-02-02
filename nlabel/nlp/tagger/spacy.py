from collections.abc import Iterable
from ..core import Builder as AbstractBuilder, Tagger, labels_from_data
from nlabel.embeddings import NativeEmbedding
from nlabel.embeddings import EmbedderFactory as AbstractEmbedderFactory

import contextlib
import logging


class Embedder:
    def prepare(self, item):
        raise NotImplementedError()

    def embedding(self, item):
        raise NotImplementedError()


class NoOpEmbedder(Embedder):
    def prepare(self, item):
        pass

    def embedding(self, item):
        pass


class NativeEmbedder(Embedder):
    def prepare(self, item):
        pass

    def embedding(self, item):
        return item.vector


class MagnitudeEmbedder(Embedder):
    def __init__(self, vectors):
        self._vectors = vectors

    def prepare(self, item):
        pass

    def embedding(self, item):
        return self._vectors.query(item.text)


class EmbedderFactory(AbstractEmbedderFactory):
    def hf(self, model, options):
        raise NotImplementedError()

    def pipeline(self):
        return NativeEmbedder()

    def flair_api(self, x):
        raise RuntimeError(
            "cannot use flair embedding API on spacy pipeline")

    def magnitude(self, vectors):
        return MagnitudeEmbedder(vectors)


@contextlib.contextmanager
def _make_embedder(vectors, name):
    if not vectors:
        yield NoOpEmbedder()
    else:
        tv = vectors.get(name)
        if tv:
            with tv.instance(EmbedderFactory()) as instance:
                yield instance
        else:
            yield NoOpEmbedder()


class Builder(AbstractBuilder):
    def __init__(self, guid, signature, doc, vectors=None, renames=None):
        super().__init__(guid, signature, (
            'sentence', 'token', 'lemma',
            'pos', 'tag', 'morph',
            'dep', 'ent_iob', 'ent'), vectors, renames)

        self._doc = doc
        self._json_data = doc.to_json()

    def add_sent(self):
        tagger = self.tagger('sentence')

        for sent in self._doc.sents:
            tagger.append({
                'start': sent.start_char,
                'end': sent.end_char
            }, vector=lambda: sent.vector)

        tagger.done()

    def add_token(self):
        tagger = self.tagger('token')

        with _make_embedder(self._vectors, 'token') as embedder:

            for token in self._doc:
                start_char = token.idx
                end_char = start_char + len(token)
                tagger.append({
                    'start': start_char,
                    'end': end_char
                }, vector=lambda: embedder.embedding(token))

        tagger.done()

    def add_tag(self, attr, split=None):
        tagger = self.tagger(attr)

        for token in self._json_data['tokens']:
            data = token.get(attr)
            if data is not None and data:
                tagger.append({
                    'start': token['start'],
                    'end': token['end'],
                    'labels': labels_from_data(data, split)
                })

        tagger.done()

    def add_ent_iob(self):
        tagger = self.tagger('ent_iob')

        for token in self._doc:
            # https://spacy.io/api/token
            if token.ent_type_:
                value = f"{token.ent_iob_}-{token.ent_type_}"
            else:
                value = f"{token.ent_iob_}"

            tagger.append({
                'start': token.idx,
                'end': token.idx + len(token.text),
                'labels': [{
                    'value': value
                }]
            })

        tagger.done()

    def add_ent(self):
        if not self._doc.ents:
            return

        tagger = self.tagger('ent')

        for ent in self._doc.ents:
            tagger.append({
                'start': ent.start_char,
                'end': ent.end_char,
                'labels': [{
                    'value': ent.label_
                }]
            }, vector=lambda: ent.vector)

        tagger.done()

    def add_dep(self):
        tagger = self.tagger('dep')

        for token in self._json_data['tokens']:
            tagger.append({
                'start': token['start'],
                'end': token['end'],
                'labels': [{
                    'value': token['dep']
                }],
                'parent': token['head']
            })

        tagger.done()


class SpacyTagger(Tagger):
    @staticmethod
    def is_compatible_nlp(nlp):
        try:
            import spacy
            return isinstance(nlp, spacy.Language)
        except ImportError:
            return False

    def __init__(self, nlp, vectors: dict = None, meta=None, renames=None, require_gpu=False):
        import spacy
        import thinc.api

        if require_gpu:
            thinc.api.set_gpu_allocator("pytorch")
            thinc.api.require_gpu()

        is_gpu = thinc.api.prefer_gpu()
        logging.info(f"thinc.api.prefer_gpu() returned {is_gpu}")

        super().__init__()
        self._renames = renames

        if not vectors:
            vectors = None
        elif isinstance(vectors, dict):
            unsupported_keys = set(vectors.keys()) - {'token', 'sentence', 'ent'}
            if unsupported_keys:
                raise ValueError(
                    f"embeddings for {unsupported_keys} are not supported")
        else:
            raise ValueError("expected vectors to be a list of names or a dict")

        self._prototype = {
            'type': 'nlp',
            'env': self._env_data(),
            'library': {
                'name': 'spacy',
                'version': spacy.__version__
            },
            'model': {
                'lang': nlp.meta['lang'],
                'name': nlp.meta['name'],
                'version': nlp.meta['version']
            }
        }

        if vectors:
            self._prototype['vectors'] = dict((k, v.to_dict()) for k, v in vectors.items())

        if renames:
            self._prototype['renames'] = renames

        if meta:
            self._prototype['meta'] = meta

        self._nlp = nlp
        self._vectors = vectors

    @property
    def signature(self):
        return self._prototype

    def _has(self, klass):
        for k, v in self._nlp.pipeline:
            if isinstance(v, klass):
                return True
        return False

    def _builder_from_doc(self, doc):
        import spacy

        builder = Builder(
            self.guid, self._prototype, doc,
            vectors=self._vectors,
            renames=self._renames)

        builder.add_sent()
        builder.add_token()

        if self._has(spacy.pipeline.Lemmatizer):
            builder.add_tag('lemma')

        if self._has(spacy.pipeline.Tagger):
            builder.add_tag('tag')

            # also present if we do not have
            # spacy.pipeline.Morphologizer:
            builder.add_tag('pos')
            builder.add_tag('morph', split="|")

        if self._has(spacy.pipeline.EntityRecognizer):
            builder.add_ent_iob()
            builder.add_ent()

        if self._has(spacy.pipeline.DependencyParser):
            builder.add_dep()

        return builder

    def process(self, text):
        return self._builder_from_doc(self._nlp(text))

    def process_n(self, texts, batch_size=1):
        for doc in self._nlp.pipe(texts, batch_size=batch_size):
            yield self._builder_from_doc(doc)
