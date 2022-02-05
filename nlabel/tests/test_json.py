from nlabel.tests import TestCase

import nlabel
import spacy


class TestDocument(TestCase):
	def test_spacy(self):
		ref_nlp = spacy.load("en_core_web_sm")
		test_nlp = nlabel.NLP(ref_nlp, vectors={'token': nlabel.embeddings.native})

		for text in self.texts:
			self._check_output(ref_nlp(text), test_nlp(text))
