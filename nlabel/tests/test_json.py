from nlabel.tests import TestCase

import nlabel
import spacy


class TestDocument(TestCase):
	def test_spacy(self):
		for lang, text in self.texts:
			ref_nlp = self.models['spacy'][lang]
			test_nlp = nlabel.NLP(ref_nlp, vectors={'token': nlabel.embeddings.native})
			self._check_output(ref_nlp(text), test_nlp(text), ref_lib='spacy')

	def test_stanza(self):
		for lang, text in self.texts:
			ref_nlp = self.models['stanza'][lang]
			test_nlp = nlabel.NLP(ref_nlp)
			self._check_output(ref_nlp(text), test_nlp(text), ref_lib='stanza')
