from nlabel.tests import TestCase
from pathlib import Path

import nlabel
import tempfile


class TestArriba(TestCase):
	def test_save_load(self):
		for lang, text in self.texts:
			ref_nlp = self.models['spacy'][lang]
			test_nlp = nlabel.NLP(ref_nlp, vectors={'token': nlabel.embeddings.native})

			with tempfile.TemporaryDirectory() as tempdir:
				path1 = Path(tempdir) / "archive1"
				path2 = Path(tempdir) / "archive2"

				with nlabel.open(path1, mode="w", engine="carenero") as archive:
					archive.add(test_nlp(text))
					archive.save(path2, engine="arriba", progress=False)

				with nlabel.open(path2, mode="r") as archive:
					self.assertEqual(archive.engine, "arriba")
					self.assertEqual(len(archive.taggers), 1)

					for doc in archive.iter(archive.taggers[0], progress=False):
						self._check_output(ref_nlp(text), doc, ref_lib='spacy')
