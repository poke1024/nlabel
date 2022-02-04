from nlabel.tests import TestCase

import nlabel
import spacy


class TestDocument(TestCase):
	def test_spacy(self):
		ref_nlp = spacy.load("en_core_web_sm")
		test_nlp = nlabel.NLP(ref_nlp)

		for text in self.texts:
			self._check_output(ref_nlp, test_nlp, text, [
				'pos', 'tag', 'dep'  #, 'vector'
			])

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
