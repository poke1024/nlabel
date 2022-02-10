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
			'ent': flair.models.SequenceTagger.load('ner-fast'),
			'pos': flair.models.SequenceTagger.load('upos-fast')})
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
					ref_data[(i, j, 'ent')] = (ent.text, ent.tag, ent.score)
				for j, pos in enumerate(sentence.get_spans('pos')):
					ref_data[(i, j, 'pos')] = (pos.text, pos.tag)

			test_data = {}
			for i, sentence in enumerate(test_nlp(text).sentences):
				for j, ent in enumerate(sentence.ents):
					test_data[(i, j, 'ent')] = (ent.text, ent.labels[0].value, ent.labels[0].score)
				for j, token in enumerate(sentence.tokens):
					test_data[(i, j, 'pos')] = (token.text, token.pos[0].value)

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

