import codecs
import collections
import functools

import orjson
import numpy as np
from numpy import searchsorted
from cached_property import cached_property
from ..common import TagError, binary_searcher


TempLabel = collections.namedtuple('TempLabel', ['value', 'score'])


class Code:
    def __init__(self, code_id, tag_form, b_code, b_data):
        self._code_id = code_id
        self._tag_form = tag_form
        self._b_code = b_code

        b_span_ids = b_data.spans.values
        w = b_span_ids.which
        if w == "none":
            self._b_span_ids = []
        else:
            self._b_span_ids = getattr(b_span_ids, str(w))

        self._b_labels = b_data.labels

        if b_data.parents.which == "none":
            self._b_parents = None
        else:
            self._b_parents = b_data.parents.indices.values

    @property
    def code_id(self):
        return self._code_id

    @property
    def name(self):
        return self._tag_form.name

    @property
    def nlp(self):
        return self._doc_data.nlps[self._b_code.tagger]

    @property
    def b_code(self):
        return self._b_code

    @property
    def form(self):
        return self._tag_form

    @cached_property
    def values(self):
        return self._b_code.values

    @property
    def b_span_ids(self):
        return self._b_span_ids

    @cached_property
    def b_span_ids_np(self):
        return np.array(self._b_span_ids, dtype=np.int32)

    @cached_property
    def b_span_id_to_index(self):
        return dict((x, i) for i, x in enumerate(self._b_span_ids))

    @cached_property
    def labels_by_span_id(self):
        b_labels = self._b_labels

        values = b_labels.values
        scores = b_labels.scores
        groups = b_labels.groups

        if values.which == "none":
            return self._tag_form.make_label, {}  # no labels

        arr_values = getattr(values, str(values.which))
        arr_scores = getattr(scores, str(scores.which)) if scores.which != 'none' else None

        if groups.which == "none":
            if arr_scores is None:
                return self._tag_form.make_label_from_value, dict(
                    (span, value)
                    for span, value in zip(self._b_span_ids, arr_values) if span >= 0)
            else:
                return self._tag_form.make_label, dict(
                    (span, [TempLabel(value, score)])
                    for span, value, score in zip(self._b_span_ids, arr_values, arr_scores) if span >= 0)
        else:
            arr_groups = getattr(groups, str(groups.which))

            if arr_scores is None:
                arr_values = list(arr_values)
                return self._tag_form.make_label_from_values, dict(
                    (span, arr_values[arr_groups[i]:arr_groups[i + 1]])
                    for span, i in zip(self._b_span_ids, range(len(arr_values)))
                    if span >= 0)

            else:
                r = {}
                for span, i in zip(self._b_span_ids, range(len(arr_values))):
                    if span >= 0:
                        r[span] = [
                            TempLabel(arr_values[j], arr_scores[j] if arr_scores else 1)
                            for j in range(arr_groups[i], arr_groups[i + 1])]

                return self._tag_form.make_label, r

    @cached_property
    def b_span_ids_searcher(self):
        return binary_searcher(self.b_span_ids_np)

    def label_for_span(self, span_id):
        make_label, b_labels_dict = self.labels_by_span_id
        b_labels = b_labels_dict.get(span_id)
        if b_labels is None:
            return self._tag_form.empty_label
        else:
            return make_label(self, b_labels)


class SpanList:
    pass


class FullSpanList(SpanList):
    def __init__(self, doc_data, code):
        self._doc_data = doc_data
        self._code = code
        self._b_span_ids = code.b_span_ids

    def __len__(self):
        return len(self._b_span_ids)

    def __getitem__(self, i):
        yield Span(self._doc_data, self._code, self._b_span_ids[i])

    def __iter__(self):
        doc_data = self._doc_data
        code = self._code
        yield from (Span(doc_data, code, x) for x in self._b_span_ids)


class SlicedSpanList(SpanList):
    def __init__(self, doc_data, code, b_container):
        b_span_ids = code.b_span_ids_np
        self._doc_data = doc_data
        self._code = code

        left_span = searchsorted(
            doc_data.starts_array,
            b_container.start)

        self._i0 = searchsorted(b_span_ids, left_span)
        self._q_end = b_container.end

        self._b_span_ids = b_span_ids

    def __len__(self):
        doc_data = self._doc_data
        b_span_ids = self._b_span_ids
        spans_array = doc_data.spans_array

        i = self._i0
        n = len(b_span_ids)
        q_end = self._q_end
        total = 0

        while i < n:
            j = b_span_ids[i]
            s_start, s_end = spans_array[j]
            if s_start >= q_end:
                break
            if s_end <= q_end:
                total += 1
            i += 1

        return total

    def __iter__(self):
        doc_data = self._doc_data
        b_span_ids = self._b_span_ids
        spans_array = doc_data.spans_array

        i = self._i0
        n = len(b_span_ids)
        q_end = self._q_end
        code = self._code

        while i < n:
            j = b_span_ids[i]
            s_start, s_end = spans_array[j]
            if s_start >= q_end:
                break
            if s_end <= q_end:
                yield Span(doc_data, code, j)
            i += 1


def unpack(x):
    return getattr(x, str(x.which))


SpanData = collections.namedtuple(
    'SpanData', ['start', 'end'])


class DocData:
    _utf8_decoder = codecs.getdecoder("utf8")

    def __init__(self, b_doc, taggers, tag_forms, vf):
        self.b_doc = b_doc
        self.taggers = taggers
        self.tag_forms = tag_forms
        self.vf = vf

    def close(self):
        self.b_doc = None
        self.taggers = None

    def decode_utf8(self, data):
        return DocData._utf8_decoder(data)[0]

    @cached_property
    def text(self):
        return self.decode_utf8(self.b_doc.text)

    def text_span(self, start, end):
        return self.decode_utf8(self.b_doc.text[start:end])

    @cached_property
    def _utf8_to_text_index(self):
        w = np.array([len(x) for x in codecs.iterencode(self.text, "utf8")], dtype=np.uint32)
        offsets = np.cumsum(np.concatenate([[0], w]))
        return dict((x, i) for i, x in enumerate(offsets))

    def span_index(self, i):
        return self._utf8_to_text_index[i]  # convert utf8 byte index to char index

    @functools.lru_cache
    def _vectors(self, code_id):
        if self.vf is None:
            return None
        else:
            return self.vf[str(code_id)]

    def vector(self, code, i):
        v = self._vectors(code.code_id)
        return None if v is None else v[i]

    def tag_form(self, name):
        form = self.tag_forms.get(name)
        if not form:
            raise TagError(name)
        return form

    @cached_property
    def spans_array(self):
        starts = np.array(unpack(self.b_doc.starts), dtype=np.uint32)
        lens = np.array(unpack(self.b_doc.lens), dtype=np.uint32)
        ends = starts + lens
        return np.column_stack((starts, ends))

    @functools.lru_cache
    def get_span(self, span_id):
        arr = self.spans_array
        return SpanData(arr[span_id, 0], arr[span_id, 1])

    @cached_property
    def starts_array(self):
        return self.spans_array[:, 0]

    def iter(self, tag, b_container=None):
        form = self.tag_form(tag)
        if form.is_plural:
            raise ValueError(
                f"use '{form.singularize().name.external}' instead of '{tag}'")

        tagger = self.taggers.get(tag)
        if tagger is None:
            raise TagError(tag)
        elif b_container:
            return SlicedSpanList(self, tagger, b_container)
        else:
            return FullSpanList(self, tagger)


class Span:
    def __init__(self, doc_data, code, b_span_id):
        assert b_span_id >= 0
        self._doc_data = doc_data
        self._code = code
        self._b_span_id = b_span_id

    @property
    def _b_doc(self):
        return self._doc_data.b_doc

    @cached_property
    def _b_span(self):
        span_id = self._b_span_id
        return self._doc_data.get_span(span_id) if span_id >= 0 else None

    @cached_property
    def start(self):
        b_span = self._b_span
        return self._doc_data.span_index(b_span.start) if b_span else None

    @cached_property
    def end(self):
        b_span = self._b_span
        return self._doc_data.span_index(b_span.end) if b_span else None

    @cached_property
    def text(self):
        b_span = self._b_span
        return self._doc_data.text_span(b_span.start, b_span.end) if b_span else None

    @property
    def vector(self):
        i = self._code.b_span_id_to_index[self._b_span_id]
        return self._doc_data.vector(self._code, i)

    @property
    def label(self):
        return self._code.label_for_span(self._b_span_id)

    def __getattr__(self, attr):
        tagger = self._doc_data.taggers.get(attr)
        if tagger is None:
            form = self._doc_data.tag_form(attr)
            if form.is_plural:  # e.g. "tokens"
                if self._b_span is None:
                    return []
                else:
                    return self._doc_data.iter(
                        form.singularize().name.external, self._b_span)
            else:
                return form.empty_label
        else:
            return tagger.label_for_span(self._b_span_id)


class Document:
    def __init__(self, doc_data):
        self._b_doc = doc_data.b_doc
        self._doc_data = doc_data

    def close(self):
        self._b_doc = None

    @property
    def __tags__(self):
        return sorted(self._doc_data.taggers.keys())

    @property
    def b_doc(self):
        return self._b_doc

    @property
    def text(self):
        return self._doc_data.text

    @cached_property
    def meta(self):
        return orjson.loads(self._b_doc.meta)

    def __getattr__(self, attr):
        form = self._doc_data.tag_form(attr)
        if form.is_plural:
            return self._doc_data.iter(
                form.singularize().name.external)
        else:
            return form.empty_label
