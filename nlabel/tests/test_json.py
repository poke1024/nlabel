from nlabel.tests import TestCase

import nlabel
import logging


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

	def test_flair(self):
		import flair
		import flair.tokenization

		ref_nlp = flair.models.MultiTagger({
			'ent': flair.models.SequenceTagger.load('ner-fast')})
		test_nlp = nlabel.NLP(ref_nlp)

		splitter = flair.tokenization.SegtokSentenceSplitter()
		for lang, text in self.texts:
			if lang != 'en':
				logging.info(f"skipping flair tests for lang {lang}")
				continue

			sentences = splitter.split(text)

			ref_data = {}
			for i, sentence in enumerate(sentences):
				ref_nlp.predict(sentence)
				for j, ent in enumerate(sentence.get_spans('ent')):
					ref_data[(i, j)] = (ent.text, ent.tag)  # ent.score

			test_data = {}
			for i, sentence in enumerate(test_nlp(text).sentences):
				for j, ent in enumerate(sentence.ents):
					test_data[(i, j)] = (ent.text, ent.label)

			self.assertEqual(ref_data, test_data)

	def test_deeppavlov(self):
		try:
			import deeppavlov
			TestCase.models['deeppavlov'] = {
				'en': deeppavlov.build_model(
					deeppavlov.configs.ner.ner_ontonotes_bert_torch, download=True)
			}
		except:
			pass

