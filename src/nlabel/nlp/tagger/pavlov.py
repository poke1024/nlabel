from nlabel.nlp.core import Builder, Tagger, labels_from_data

import logging


def _derive_spans(text, tokens):
    i0 = 0
    for s in tokens:
        i = text.index(s, i0)
        yield i, i + len(s)
        i0 = i + len(s)


class EntBuilder(Builder):
    def __init__(self, prototype, model, sents, renames=None):
        super().__init__(prototype, renames=renames)
        self._model = model
        self._sents = sents

    def add_ent(self):
        tagger = self.tagger('ent_iob')

        for i, (tokens, tags) in enumerate(zip(*self._model(self._sents))):
            spans = list(_derive_spans(self._sents[i], tokens))
            for (start, end), tag in zip(spans, tags):
                tagger.append({
                    'start': start,
                    'end': end,
                    'labels': [{
                        'value': tag
                    }]
                })


class MorphBuilder(Builder):
    def __init__(self, prototype, model, sents, renames=None):
        super().__init__(prototype, renames=renames)
        self._model = model
        self._sents = sents

        for x in self._model.pipe:
            item = x[-1]
            if isinstance(item, deeppavlov.models.morpho_tagger.common.TagOutputPrettifier):
                item.set_format_mode('ud')

        ud_keys = ('id', 'word', 'lemma', 'pos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc')

        self._sents = []
        for i, res in enumerate(self._model(self._sents)):
            records = []
            for item in res.strip("\n").split("\n"):
                records.append(dict((k, v) for k, v in zip(ud_keys, item.split()) if v != '_'))

            spans = list(_derive_spans(self._sents[i], [x['word'] for x in records]))
            for (start, end), record in zip(spans, records):
                record['start'] = start
                record['end'] = end

            self._sents.append(records)

    def add_tag(self, attr, split=None):
        if not any(any(attr in token for token in sent) for sent in self._sents):
            return

        tagger = self.tagger(attr)
        for sent in self._sents:
            for token in sent:
                labels = labels_from_data(token.get(attr), split)
                tagger.append({
                    'start': token['start'],
                    'end': token['end'],
                }.update(dict([('labels', labels)] if labels else [])))


class PavlovTagger(Tagger):
    @staticmethod
    def is_compatible_nlp(nlp):
        try:
            import deeppavlov
            return isinstance(nlp, deeppavlov.core.common.chainer.Chainer)
        except ImportError:
            return False

    def __init__(self, nlp, kind, sentencizer, vectors=False, meta=None, renames=None, require_gpu=False):
        import deeppavlov
        import spacy

        if vectors is not False:
            raise RuntimeError("vectors are not supported for deeppavlov")

        if require_gpu:
            logging.warning("require_gpu was ignored.")

        self._model = nlp
        self._kind = kind
        self._renames = renames

        self._prototype = {
            'type': 'nlp',
            'env': self._env_data(),
            'library': {
                'name': 'deeppavlov',
                'version': deeppavlov.__version__
            },
            'sentencizer': {
                'library': {
                    'name': 'spacy',
                    'version': spacy.__version__
                },
                'model': {
                    'lang': sentencizer.meta['lang'],
                    'name': sentencizer.meta['name'],
                    'version': sentencizer.meta['version']
                }
            }
        }

        sentencizer.disable_pipes(
            *[x for x in sentencizer.pipe_names if x != 'sentencizer'])
        self._sentencizer = sentencizer

        if renames:
            self._prototype['renames'] = renames

        if meta:
            self._prototype['meta'] = meta

    @property
    def description(self):
        return self._prototype

    def process(self, text):
        sents = []
        for sent in self._sentencizer(text).sents:
            sents.append(sent.sent.string.strip())

        if self._kind == 'ent':
            builder = EntBuilder(
                self._prototype, self._model, sents,
                renames=self._renames)
            builder.add_ent()
            return builder
        elif self._kind == 'morph':
            builder = MorphBuilder(
                self._prototype, self._model, sents,
                renames=self._renames)
            builder.add_tag('token')
            builder.add_tag('lemma')
            builder.add_tag('pos')
            builder.add_tag('xpos')
            builder.add_tag('feats', split="|")
            return builder
        else:
            raise RuntimeError(f"illegal model kind {self._kind}")
