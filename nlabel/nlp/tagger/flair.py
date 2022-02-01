import contextlib

from nlabel.nlp.core import Builder, Tagger
from nlabel.embeddings import EmbedderFactory as AbstractEmbedderFactory, Embedding

import logging


class FlairModelName:
    def __init__(self, name):
        parts = name.split("-")

        if len(parts[0]) == 2:
            lang = parts[0]
            parts = parts[1:]
        else:
            lang = 'en'

        self._lang = lang
        self._kind = parts[0]

    @property
    def lang(self):
        return self._lang

    @property
    def kind(self):
        return self._kind


class Embedder:
    def prepare(self, sentence):
        raise NotImplementedError()

    def token_embedding(self, token):
        raise NotImplementedError()


class NoOpEmbedder(Embedder):
    def prepare(self, sentence):
        pass

    def token_embedding(self, token):
        pass


class FlairEmbedder(Embedder):
    def __init__(self, embedding):
        self._embedding = embedding

    def prepare(self, sentence):
        self._embedding.embed(sentence)

    def token_embedding(self, token):
        return token.embedding


class MagnitudeEmbedder(Embedder):
    def __init__(self, vectors):
        self._vectors = vectors

    def prepare(self, sentence):
        pass

    def token_embedding(self, token):
        return self._vectors.query(token.text)


class EmbedderFactory(AbstractEmbedderFactory):
    def hf(self, model, options):
        from flair.embeddings import TransformerWordEmbeddings
        return FlairEmbedder(TransformerWordEmbeddings(model, **options))

    def pipeline(self):
        raise RuntimeError("flair does not have native pipeline embeddings")

    def flair_api(self, x):
        return FlairEmbedder(x)

    def magnitude(self, vectors):
        return MagnitudeEmbedder(vectors)


@contextlib.contextmanager
def _make_embedder(vectors):
    if not vectors:
        yield NoOpEmbedder()
    else:
        tv = vectors.get('token')
        if tv:
            with tv.instance(EmbedderFactory()) as instance:
                yield instance
        else:
            yield NoOpEmbedder()


class FlairBuilder(Builder):
    def __init__(self, prototype, sents, vectors=None, renames=None):
        super().__init__(prototype, vectors=vectors, renames=renames)

        self._sents = sents

    def add_sentences_and_tokens(self):
        sentence_tagger = self.tagger('sentence')
        token_tagger = self.tagger('token')

        with _make_embedder(self._vectors) as embedder:

            for sentence in self._sents:
                sentence_tagger.append({
                    'start': sentence.start_pos,
                    'end': sentence.end_pos
                })

                embedder.prepare(sentence)

                for token in sentence:
                    token_tagger.append({
                        'start': sentence.start_pos + token.start_pos,
                        'end': sentence.start_pos + token.end_pos,
                    }, vector=lambda: embedder.token_embedding(token))

        sentence_tagger.done()
        token_tagger.done()

    def add_tags(self, sentence):
        label_types = []
        for token in sentence:
            for annotation in token.annotation_layers.keys():
                if annotation not in label_types:
                    label_types.append(annotation)

        for label_type in label_types:
            tagger = self.tagger(label_type, force_empty=False)

            data = [x.to_dict() for x in sentence.get_spans(label_type)]
            for record in data:
                labels = []
                for label in record['labels']:
                    labels.append({
                        'value': label.value,
                        'score': label.score
                    })
                labels = sorted(
                    labels, key=lambda x: x['score'], reverse=True)

                tagger.append({
                    'start': sentence.start_pos + record['start_pos'],
                    'end': sentence.start_pos + record['end_pos'],
                    'labels': labels
                })

            tagger.done()


def _create_sentence_splitter(sentence_splitter, from_spacy):
    import flair.tokenization

    if from_spacy:
        try:
            import spacy
            if not isinstance(from_spacy, spacy.Language):
                raise ValueError(f"expected spacy.Language, got {from_spacy}")
            if sentence_splitter is not None:
                raise ValueError("cannot specify both from_spacy and sentence_splitter")
            return flair.tokenization.SpacySentenceSplitter(
                from_spacy, tokenizer=flair.tokenization.SpacyTokenizer(from_spacy))
        except ImportError:
            pass

    if sentence_splitter is None:
        sentence_splitter = flair.tokenization.SegtokSentenceSplitter()

    if not isinstance(sentence_splitter, flair.tokenization.SentenceSplitter):
        raise ValueError(
            f"expected flair.tokenization.SentenceSplitter, "
            f"got {sentence_splitter}")

    return sentence_splitter


def _create_vectors(vectors):
    if vectors:
        if isinstance(vectors, dict):
            unsupported_keys = set(vectors.keys()) - {'token'}
            if unsupported_keys:
                raise ValueError(
                    f"embeddings for {unsupported_keys} are not supported")
        elif isinstance(vectors, Embedding):
            vectors = {
                'token': vectors
            }
        else:
            raise ValueError(f"expected nlabel.vectors.Embedding, got {vectors}")

        with vectors['token'].instance(EmbedderFactory()) as _:
            pass  # fail early if wrong args

    return vectors


def _detect_tagger_lang(tagger):
    flair_tagger_names = tagger.name_to_tagger.keys()
    langs = [FlairModelName(x).lang for x in flair_tagger_names]
    if langs:
        lang = langs[0]
        if not all(x == lang for x in langs):
            raise ValueError(f"detected inconsistent lang codes: {langs}")
    else:
        lang = 'unknown'
    return lang


class FlairTagger(Tagger):
    @staticmethod
    def is_compatible_nlp(nlp):
        try:
            import flair.models
            return isinstance(nlp, flair.models.MultiTagger)
        except ImportError:
            return False

    def __init__(
        self, tagger, from_spacy=None, sentence_splitter=None, vectors: Embedding = False,
        meta=None, renames=None, require_gpu=False):

        import flair

        import torch
        cuda_available = torch.cuda.is_available()
        logging.info(f"torch.cuda.is_available() returned {cuda_available}")
        if require_gpu and not cuda_available:
            raise RuntimeError("require_gpu is True, but cuda is not available")

        super().__init__()

        self._vectors = _create_vectors(vectors)

        sentence_splitter = _create_sentence_splitter(sentence_splitter, from_spacy)

        self._prototype = {
            'type': 'nlp',
            'env': self._env_data(),
            'library': {
                'name': 'flair',
                'version': flair.__version__
            },
            'model': {
                'lang': _detect_tagger_lang(tagger),
                'taggers': sorted(tagger.name_to_tagger.keys()),
                'sentence_splitter': sentence_splitter.name
            }
        }

        if self._vectors:
            self._prototype['vectors'] = dict((k, v.to_dict()) for k, v in self._vectors.items())

        if renames:
            self._prototype['renames'] = renames

        if meta:
            self._prototype['meta'] = meta

        self._nlp = tagger
        self._sentence_splitter = sentence_splitter
        self._renames = renames

    @property
    def description(self):
        return self._prototype

    def _split_sents(self, text):
        # stripping whitespace from the right suppresses
        # "Warning: An empty Sentence was created!"
        # from flair's Sentence constructor in flair/data.py.

        # rstrip() will not modify our text indices, so it
        # will be ok, apart from having no tags for the
        # stripped whitespace.

        return self._sentence_splitter.split(text.rstrip())

    def process(self, text):
        sents = self._split_sents(text)

        builder = FlairBuilder(
            self._prototype, sents,
            vectors=self._vectors,
            renames=self._renames)

        builder.add_sentences_and_tokens()

        for sentence in sents:
            self._nlp.predict(sentence)
            builder.add_tags(sentence)

        return builder
