import contextlib
from pathlib import Path


class EmbedderFactory:
    def hf(self, model, options):
        raise NotImplementedError()

    def pipeline(self):
        raise NotImplementedError()

    def flair_api(self, x):
        raise NotImplementedError()

    def magnitude(self, vectors):
        raise NotImplementedError()


class Embedding:
    @contextlib.contextmanager
    def instance(self, factory: EmbedderFactory):
        raise NotImplementedError()

    def to_dict(self):
        raise NotImplementedError()


class NativeEmbedding(Embedding):
    @contextlib.contextmanager
    def instance(self, factory: EmbedderFactory):
        yield factory.pipeline()

    def to_dict(self):
        return dict(
            type='native')


class MagnitudeEmbedding(Embedding):
    def __init__(self, path):
        self._path = Path(path)

    @contextlib.contextmanager
    def instance(self, factory: EmbedderFactory):
        import pymagnitude

        vectors = pymagnitude.Magnitude(self._path)
        try:
            yield factory.magnitude(vectors)
        finally:
            vectors.close()

    def to_dict(self):
        return dict(
            type='magnitude',
            name=self._path.name)


class HuggingFaceEmbedding(Embedding):
    def __init__(self, model, **kwargs):
        self._model = model
        self._kwargs = kwargs

    @contextlib.contextmanager
    def instance(self, factory: EmbedderFactory):
        yield factory.hf(self._model, self._kwargs)

    def to_dict(self):
        return dict(
            type='huggingface',
            model=self._model,
            **self._kwargs)


class FlairAPIEmbedding(Embedding):
    def __init__(self, embedding):
        self._embedding = embedding

    @contextlib.contextmanager
    def instance(self, factory: EmbedderFactory):
        yield factory.flair_api(self._embedding)

    def to_dict(self):
        return dict(
            type='flair_api',
            flair_type=self._embedding.embedding_type,
            flair_names=self._embedding.get_names())


native = NativeEmbedding()


def floret():
    pass


def magnitude(path):
    return MagnitudeEmbedding(path)


def huggingface(model, **kwargs):
    return HuggingFaceEmbedding(model, **kwargs)


def flair_api(embedding):
    from flair.embeddings import Embeddings
    if not isinstance(embedding, Embeddings):
        raise ValueError(
            f"expected flair.embeddings.Embeddings, got {embedding}")
    return FlairAPIEmbedding(embedding)
