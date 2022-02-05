import unittest
import collections
import json
import spacy
import stanza


class Extractor:
    def __init__(self, ref_lib):
        self._ref_lib = ref_lib

    def __call__(self, doc):
        return getattr(self, self._ref_lib)(doc)

    def spacy(self, doc):
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
                yield 'ent', (i, j, (ent.text, ent.label_))

    def stanza(self, doc):
        for i, sentence in enumerate(doc.sentences):
            for j, word in enumerate(sentence.words):
                yield 'text', (i, j, word.text)
                yield 'lemma', (i, j, word.lemma)
                yield 'upos', (i, j, word.upos)
                yield 'xpos', (i, j, word.xpos)
                yield 'feats', (i, j, (word.feats.split("|") if word.feats else []))
                yield 'dep', (i, j, word.deprel)
                yield 'head', (i, j, (sentence.words[word.head - 1].text if word.head > 0 else word.text))

            for j, ent in enumerate(sentence.ents):
                yield 'ent', (i, j, (ent.text, ent.type))


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
    models = None

    @classmethod
    def setUpClass(cls):
        if TestCase.models is None:
            TestCase.models = {
                'spacy': {
                    'en': spacy.load('en_core_web_sm'),
                    'ja': spacy.load('ja_core_news_sm')
                },
                'stanza': {
                    'en': stanza.Pipeline('en', verbose=False),
                    'ja': stanza.Pipeline('ja', verbose=False)
                }
            }

    @property
    def texts(self):
        with open("texts.json", "r") as f:
            for lang, texts in json.loads(f.read()).items():
                for text in texts:
                    yield lang, text

    def _nlabel_data(self, doc, attr):
        data = collections.defaultdict(dict)

        for i, sentence in enumerate(doc.sentences):
            if attr == 'head':
                for j, dep in enumerate(sentence.deps):
                    yield i, j, dep.parent.text
            elif attr == 'ent':
                for j, ent in enumerate(sentence.ents):
                    yield i, j, (ent.text, ent.label)
            elif attr == 'vector':
                for j, token in enumerate(sentence.tokens):
                    yield i, j, tuple(token.vector)
            else:
                for j, token in enumerate(sentence.tokens):
                    yield i, j, getattr(token, attr)

        return data

    def _check_output(self, ref_doc, test_doc, ref_lib):
        ref_extractor = Extractor(ref_lib)
        ref_data = gather_by_key(ref_extractor(ref_doc))

        for attr, ref_attr_data in ref_data.items():
            with self.subTest(tag=attr):
                self.assertEqual(
                    ref_attr_data,
                    list(self._nlabel_data(test_doc, attr)))

        self.assertEqual(
            tuple(self._nlabel_data(test_doc, 'text')),
            tuple(_nlabel_data_iter(test_doc, 'text')))

        for i, sentence in enumerate(test_doc.sentences):
            for j, token in enumerate(sentence.tokens):
                self.assertEqual(token.text, test_doc.text[token.start:token.end])
            for j, ent in enumerate(sentence.ents):
                self.assertEqual(ent.text, test_doc.text[ent.start:ent.end])
