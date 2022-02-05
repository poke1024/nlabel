from nlabel.tests import TestCase
from pathlib import Path

import nlabel
import spacy
import tempfile


class TestCarenero(TestCase):
	def test_save_load(self):
		ref_nlp = spacy.load("en_core_web_sm")
		test_nlp = nlabel.NLP(ref_nlp, vectors={'token': nlabel.embeddings.native})

		for text in self.texts:
			with tempfile.TemporaryDirectory() as tempdir:
				path = Path(tempdir) / "archive"

				with nlabel.open(path, mode="w", engine="carenero") as archive:
					archive.add(test_nlp(text))

				with nlabel.open(path, mode="r", engine="carenero") as archive:
					self.assertEqual(len(archive.taggers), 1)

					for doc in archive.iter(archive.taggers[0], progress=False):
						self._check_output(ref_nlp(text), doc)
