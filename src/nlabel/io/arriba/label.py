class Label:
    def __init__(self, tagger, value, score=None):
        self._tagger = tagger
        self._value = value
        self._score = score

    @property
    def nlp(self):
        return self._tagger.nlp

    @property
    def value(self):
        return self._value

    @property
    def score(self):
        return self._score


class LabelFactory:
    @property
    def empty_label(self):
        raise NotImplementedError()

    def make_label(self, tagger, labels):
        raise NotImplementedError()

    def make_label_from_value(self, tagger, value):
        raise NotImplementedError()

    def make_label_from_values(self, tagger, values):
        raise NotImplementedError()


class StrLabelFactory(LabelFactory):
    @property
    def empty_label(self):
        return ''

    def make_label(self, tagger, labels):
        if not labels:
            return ''
        else:
            return tagger.values[labels[0].value]

    def make_label_from_value(self, tagger, value):
        return tagger.values[value]

    def make_label_from_values(self, tagger, values):
        return tagger.values[values[0]]


class StrsLabelFactory(LabelFactory):
    @property
    def empty_label(self):
        return []

    def make_label(self, tagger, labels):
        values = tagger.values
        return [values[x.value] for x in labels]

    def make_label_from_value(self, tagger, value):
        return [tagger.values[value]]

    def make_label_from_values(self, tagger, values):
        return [tagger.values[x] for x in values]


class SingleLabelFactory(LabelFactory):
    @property
    def empty_label(self):
        return None

    def make_label(self, tagger, labels):
        if not labels:
            return None
        else:
            if len(labels) != 1:
                raise ValueError(
                    f"cannot reduce {tagger.name} labels {labels} to single label")
            return Label(tagger, tagger.values[labels[0].value], labels[0].score)

    def make_label_from_value(self, tagger, value):
        return Label(tagger, tagger.values[value])

    def make_label_from_values(self, tagger, values):
        return Label(tagger, tagger.values[values[0]])


class MultiLabelFactory(LabelFactory):
    @property
    def empty_label(self):
        return []

    def make_label(self, tagger, labels):
        return [Label(tagger, tagger.values[x.value], x.score) for x in labels]

    def make_label_from_value(self, tagger, value):
        return [Label(tagger, tagger.values[value])]

    def make_label_from_values(self, tagger, values):
        return [Label(tagger, tagger.values[x]) for x in values]


factories = {
    'str': StrLabelFactory(),
    'strs': StrsLabelFactory(),
    'label': SingleLabelFactory(),
    'labels': MultiLabelFactory()
}
