from nlabel.nlp.core import Builder, Tagger, labels_from_data

import logging


class StanzaBuilder(Builder):
    def __init__(self, prototype, doc, renames=None):
        super().__init__(prototype, (
            'sentence', 'token', 'lemma',
            'upos', 'xpos', 'feats',
            'dep', 'ent_bioes', 'ent'), renames=renames)

        self._doc = doc

    def add_sent(self):
        tagger = self.tagger('sentence')

        for sent in self._doc.sentences:
            tagger.append({
                'start': sent.tokens[0].start_char,
                'end': sent.tokens[-1].end_char
            })

        tagger.done()

    def add_token(self):
        tagger = self.tagger('token')

        for sent in self._doc.sentences:
            for token in sent.tokens:
                tagger.append({
                    'start': token.start_char,
                    'end': token.end_char
                })

        tagger.done()

    def add_tag(self, attr, rename=None, split=None):
        tagger = self.tagger(rename if rename else attr)

        for i, sent in enumerate(self._doc.sentences):
            for token in sent.tokens:
                if len(token.words) == 1:
                    data = token.words[0].to_dict().get(attr)
                    tagger.append({
                        'start': token.start_char,
                        'end': token.end_char,
                        'labels': labels_from_data(data, split)
                    })
                else:
                    parent_i = len(tagger)
                    tagger.append({
                        'start': token.start_char,
                        'end': token.end_char
                    })

                    for word in token.words:
                        data = word.to_dict().get(attr)
                        tagger.append({
                            'parent': parent_i,
                            'labels': labels_from_data(data, split)
                        })

        tagger.done()

    def add_ent(self):
        tagger = self.tagger('ent')

        for sent in self._doc.sentences:
            for ent in sent.ents:
                tagger.append({
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'labels': [{
                        'value': ent.type
                    }]
                })

        tagger.done()

    def add_dep(self):
        tagger = self.tagger('dep')

        for sent in self._doc.sentences:
            offset = len(tagger)
            for i, word in enumerate(sent.words):
                if word.head > 0:
                    parent = word.head - 1 + offset
                else:
                    parent = i + offset  # i.e. ROOT
                tagger.append({
                    'start': word.start_char,
                    'end': word.end_char,
                    'labels': [{
                        'value': word.deprel
                    }],
                    'parent': parent
                })

        tagger.done()


class StanzaTagger(Tagger):
    @staticmethod
    def is_compatible_nlp(nlp):
        try:
            import stanza
            return isinstance(nlp, stanza.Pipeline)
        except ImportError:
            return False

    def __init__(self, nlp, vectors=False, meta=None, renames=None, require_gpu=False):
        import stanza

        super().__init__()

        if vectors is not False:
            raise RuntimeError("vectors are not supported for stanza")

        if require_gpu:
            logging.warning("require_gpu was ignored.")

        self._prototype = {
            'type': 'nlp',
            'env': self._env_data(),
            'library': {
                'name': 'stanza',
                'version': stanza.__version__
            },
            'model': {
                'lang': nlp.lang
            }
        }

        if renames:
            self._prototype['renames'] = renames

        if meta:
            self._prototype['meta'] = meta

        self._nlp = nlp
        self._renames = renames

    @property
    def description(self):
        return self._prototype

    def process(self, text):
        doc = self._nlp(text)

        builder = StanzaBuilder(
            self._prototype, doc, renames=self._renames)

        builder.add_sent()
        builder.add_token()
        builder.add_tag('lemma')
        builder.add_tag('upos')
        builder.add_tag('xpos')
        builder.add_tag('feats', split="|")

        builder.add_tag('ner', rename='ent_bioes')
        builder.add_ent()
        builder.add_dep()

        return builder
