from nlabel.tests import TestCase

import nlabel
import spacy


class TestDocument(TestCase):
	def test_spacy(self):
		ref_nlp = spacy.load("en_core_web_sm")
		test_nlp = nlabel.NLP(ref_nlp)

		for text in self.texts:
			self._check_output(ref_nlp, test_nlp, text)

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

	def test_spacy_vectors(self):
		if False:
			ref_nlp = spacy.load("en_core_web_sm")

			test_nlp = nlabel.NLP(
				ref_nlp,
				vectors={'token': nlabel.embeddings.native})

			for text in self.texts:
				self._check_output(ref_nlp, test_nlp, text, [
					'vector'
				])

	def test_label_strs(self):
		pass

	def test_renames(self):
		pass
