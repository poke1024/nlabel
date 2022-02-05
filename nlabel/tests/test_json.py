from nlabel.tests import TestCase

import nlabel
import spacy


class TestDocument(TestCase):
	def test_spacy(self):
		ref_nlp = spacy.load("en_core_web_sm")
		test_nlp = nlabel.NLP(ref_nlp, vectors={'token': nlabel.embeddings.native})

		for text in self.texts:
			self._check_output(ref_nlp(text), test_nlp(text))

	def test_start_end(self):
		ref_nlp = spacy.load("en_core_web_sm")
		test_nlp = nlabel.NLP(ref_nlp)

		for text in self.texts:
			doc = test_nlp(text)
			for i, sentence in enumerate(doc.sentences):
				for j, token in enumerate(sentence.tokens):
					self.assertEqual(token.text, doc.text[token.start:token.end])
				for j, ent in enumerate(sentence.ents):
					self.assertEqual(ent.text, doc.text[ent.start:ent.end])
