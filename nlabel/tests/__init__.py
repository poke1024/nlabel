import unittest
import collections
import json


def _spacy_data(doc):
    data = collections.defaultdict(dict)

    for i, sentence in enumerate(doc.sents):
        for j, token in enumerate(sentence):
            yield 'text', (i, j, token.text)
            yield 'lemma', (i, j, token.lemma_)
            yield 'tag', (i, j, token.tag_)
            yield 'pos', (i, j, token.pos_)
            yield 'morph', (i, j, (str(token.morph).split("|") if str(token.morph) else []))
            yield 'dep', (i, j, token.dep_)
            yield 'head', (i, j, token.head.text)
            yield 'vector', (i, j, tuple(token.vector))

        for j, ent in enumerate(sentence.ents):
            yield 'ent', (i, j, ent.text)

    return data


def gather_by_key(x):
    g = collections.defaultdict(list)
    for k, v in x:
        g[k].append(v)
    return g


def _nlabel_data_iter(doc, attr):
    data = collections.defaultdict(dict)

    for i, sentence in enumerate(doc.iter('sentence')):
        for j, token in enumerate(sentence.iter('token')):
            yield i, j, getattr(token, attr)

    return data


class TestCase(unittest.TestCase):
    @property
    def texts(self):
        with open("texts.json", "r") as f:
            for text in json.loads(f.read()):
                yield text

    def _nlabel_data(self, doc, attr):
        data = collections.defaultdict(dict)

        for i, sentence in enumerate(doc.sentences):
            if attr == 'head':
                for j, dep in enumerate(sentence.deps):
                    yield i, j, dep.parent.text
            elif attr == 'ent':
                for j, ent in enumerate(sentence.ents):
                    yield i, j, ent.text
            elif attr == 'vector':
                for j, token in enumerate(sentence.tokens):
                    yield i, j, tuple(token.vector)
            else:
                for j, token in enumerate(sentence.tokens):
                    yield i, j, getattr(token, attr)

        return data

    def _check_output(self, ref_doc, test_doc):
        ref_data = gather_by_key(_spacy_data(ref_doc))

        for attr, ref_attr_data in ref_data.items():
            with self.subTest(tag=attr):
                self.assertEqual(
                    ref_attr_data,
                    list(self._nlabel_data(test_doc, attr)))

        self.assertEqual(
            tuple(self._nlabel_data(test_doc, 'text')),
            tuple(_nlabel_data_iter(test_doc, 'text')))
