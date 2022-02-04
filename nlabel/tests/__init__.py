import unittest
import collections
import json


def _gen_data_spacy(doc, attr):
    data = collections.defaultdict(dict)

    for i, sentence in enumerate(doc.sents):
        for j, token in enumerate(sentence):
            yield 'text', (i, j, token.text)
            yield 'pos', (i, j, token.pos_)
            yield 'tag', (i, j, token.tag_)
            yield 'dep', (i, j, token.dep_)

            data[i][j] = getattr(token, f'{attr}_')

    return data


def _gen_spacy_ent(doc):
    data = collections.defaultdict(dict)

    for i, sentence in enumerate(doc.sents):
        for j, ent in enumerate(sentence.ents):
            data[i][j] = ent.text

    return data


def _gen_data_nlabel(doc, attr):
    data = collections.defaultdict(dict)

    for i, sentence in enumerate(doc.sentences):
        for j, token in enumerate(sentence.tokens):
            data[i][j] = getattr(token, attr)

    return data


def _gen_data_nlabel_iter(doc, attr):
    data = collections.defaultdict(dict)

    for i, sentence in enumerate(doc.iter('sentence')):
        for j, token in enumerate(sentence.iter('token')):
            data[i][j] = getattr(token, attr)

    return data


def _gen_nlabel_ent(doc):
    data = collections.defaultdict(dict)

    for i, sentence in enumerate(doc.sentences):
        for j, ent in enumerate(sentence.ents):
            data[i][j] = ent.text

    return data


def _gen_nlabel_ent_2(doc):
    data = collections.defaultdict(dict)

    for i, sentence in enumerate(doc.sentences):
        for j, ent in enumerate(sentence.ents):
            data[i][j] = doc.text[ent.start:ent.end]

    return data


class TestCase(unittest.TestCase):
    @property
    def texts(self):
        with open("texts.json", "r") as f:
            for text in json.loads(f.read()):
                yield text

    def _check_output(self, ref_nlp, test_nlp, text, attributes):
        ref_doc = ref_nlp(text)
        test_doc = test_nlp(text)

        for attr in attributes:
            ref_data = _gen_data_spacy(ref_doc, attr)

            self.assertEqual(
                ref_data, _gen_data_nlabel(test_doc, attr))

            self.assertEqual(
                ref_data, _gen_data_nlabel_iter(test_doc, attr))

        self.assertEqual(_gen_spacy_ent(ref_doc), _gen_nlabel_ent(test_doc))
        self.assertEqual(_gen_spacy_ent(ref_doc), _gen_nlabel_ent_2(test_doc))
