import contextlib
import itertools
import functools
import numpy as np
import logging

from cached_property import cached_property
from numpy import searchsorted
from nlabel.io.common import AbstractSpanFactory, TagError
from nlabel.io.selector import make_selector
from nlabel.io.form import inflected_tag_forms


class Document:
    def __init__(self, group):
        self._group = group
        self._text = group.text
        self._spans = None
        self._tag_spans = None
        self._tag_forms = None

    @staticmethod
    @contextlib.contextmanager
    def open(path, *selectors, vectors=True):
        from .group import Group

        with Group.open(path, vectors=vectors) as group:
            yield group.view(*selectors)

    def save(self, path, exist_ok=False):
        self._group.save(path, exist_ok=exist_ok)

    @cached_property
    def _tags(self):
        tags = []
        for x in self._tag_forms.values():
            if not x.is_plural:
                tags.append(x.tag)
        return sorted(tags, key=lambda x: x.name)

    @property
    def group(self):
        return self._group

    @property
    def _external_key(self):
        return self._group.external_key

    @property
    def text(self):
        return self._text

    @property
    def _meta(self):
        return self._group.meta

    def _late_init(self, spans, tag_spans, tag_forms):
        self._spans = spans
        self._tag_spans = tag_spans
        self._tag_forms = inflected_tag_forms(tag_forms)

    def _tag_form(self, name):
        form = self._tag_forms.get(name)
        if not form:
            raise TagError(name, list(self._tag_forms.keys()))
        return form

    @functools.lru_cache
    def _starts(self, tag):
        return np.array([x[1].start for x in self._tag_spans[tag].sorted_spans], dtype=np.int32)

    def iter(self, tag, container=None):
        form = self._tag_form(tag)
        if form.is_plural:
            raise ValueError(
                f"use '{form.singularize().name}' instead of '{tag}'")
        return self._iter(form, container)

    def _iter(self, form, container=None):
        tag = form.name.external
        tag_data = self._tag_spans.get(tag)
        if tag_data is None:
            return form.make_empty_label('labels')
        elif container:
            start = container.start
            end = container.end

            i0 = searchsorted(
                self._starts(tag),
                start)

            spans = filter(lambda x: x[1].end <= end, itertools.takewhile(
                lambda x: x[1].start < end, tag_data.sorted_spans[i0:]))

            return [Tag(tag_data, i) for i, _ in spans]
        else:
            return [Tag(tag_data, i) for i, _ in tag_data.sorted_spans]

    def __getattr__(self, attr):
        form = self._tag_form(attr)
        if form.is_plural:
            return self._iter(form.singularize())
        else:
            return form.empty_label


class Label:
    __slots__ = 'value', 'score', 'nlp'

    def __init__(self, value, score=None, nlp=None):
        self.value = value
        self.score = score
        self.nlp = nlp


class TagData:
    def __init__(self, view, producer_index, name, spans, parents):
        self._view = view
        self._producer_index = producer_index
        self._name = name
        self._spans = spans  # natural order in document file
        self._parents = parents
        vg = view.group.vectors.get(producer_index)
        self._vectors = vg.get(name.internal) if vg else None

    @property
    def vectors(self):
        return self._vectors

    @property
    def name(self):
        return self._name

    @property
    def spans(self):
        return self._spans

    @property
    def parents(self):
        return self._parents

    @cached_property
    def sorted_spans(self):
        return sorted(enumerate(self._spans), key=lambda x: x[1].index)


class Tag:
    def __init__(self, tag_data, index):
        self._tag_data = tag_data
        self._index = index
        self._span = tag_data.spans[index]

    @property
    def name(self):
        return self._tag_data.name.external

    @property
    def label(self):
        return self._span.get_attr(
            self.name, kind="label")

    @property
    def labels(self):
        return self._span.get_attr(
            self.name, kind="labels")

    @property
    def start(self):
        return self._span.start

    @property
    def end(self):
        return self._span.end

    @property
    def text(self):
        return self._span.text

    @property
    def parent(self):
        i = self._tag_data.parents[self._index]
        if i is None:
            return None
        else:
            return Tag(self._tag_data, i)

    @property
    def vector(self):
        vectors = self._tag_data.vectors
        return vectors[self._index] if vectors is not None else None

    def iter(self, tag):
        return self._span.iter(tag)

    def __getattr__(self, attr):
        return self._span.get_attr(attr, kind="default")


class Span:
    __slots__ = '_view', '_start', '_end', '_index', '_tags_data'

    def __init__(self, view, start, end):
        self._view = view
        self._start = start
        self._end = end
        self._index = None
        self._tags_data = {}

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        assert self._index is None
        self._index = index

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    def add_labels(self, tagger_index, tag, data):
        if not data:
            return

        # FIXME. pass this on.
        #nlp = self._view.group.taggers[tagger_index]

        assert tag not in self._tags_data
        self._tags_data[tag] = data

    @property
    def text(self):
        return self._view.text[self._start:self._end]

    def iter(self, tag):
        return self._view.iter(tag, self)

    def get_attr(self, attr, kind):
        form = self._view._tag_form(attr)
        if form.is_plural:  # e.g. "tokens"
            return self._view.iter(
                form.singularize().name.external, self)
        else:
            data = self._tags_data.get(attr)
            if data is None:
                return form.make_empty_label(kind)
            else:
                return form.make_label(data, kind)


class SpanFactory(AbstractSpanFactory):
    def __init__(self, view):
        super().__init__()
        self._view = view

    def _make_span(self, start, end):
        return Span(self._view, start, end)


class ViewBuilder:
    def __init__(self, doc):
        self._view = Document(doc)
        self._tag_spans = {}
        self._span_factory = SpanFactory(self._view)

    def add(self, producer_index, form, tags):
        name = form.name
        span_factory = self._span_factory
        spans = []
        parents = []

        for tag in tags:
            start = tag.get('start')
            end = tag.get('end')
            if start is not None and end is not None:
                span = span_factory.get(
                    start, end)
                span.add_labels(
                    producer_index,
                    name.external,
                    tag.get('data'))
                spans.append(span)
                parents.append(tag.get('parent'))
            else:
                pass  # a sub tag from stanza, FIXME

        self._tag_spans[name.external] = TagData(
            self._view, producer_index, name, spans, parents)

    def make_view(self, tag_forms):
        spans = self._span_factory.sorted_spans

        for i, span in enumerate(spans):
            span.index = i

        self._view._late_init(
            spans, self._tag_spans, tag_forms)

        return self._view


class Loader:
    def __init__(self, *selectors, inherit_labels=True):
        self._selector = make_selector(selectors)

    def __call__(self, group):
        builder = ViewBuilder(group)
        tag_forms = self._selector.build(
            [x._.data for x in group.taggers], builder.add)
        return builder.make_view(tag_forms)
