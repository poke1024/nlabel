class LabelFactory:
    @property
    def empty_label(self):
        raise NotImplementedError()

    def make_label(self, labels):
        raise NotImplementedError()


class StrLabelFactory(LabelFactory):
    @property
    def empty_label(self):
        return ''

    def make_label(self, labels):
        if not labels:
            return ''
        else:
            return '|'.join(x.value for x in labels)


class StrsLabelFactory(LabelFactory):
    @property
    def empty_label(self):
        return []

    def make_label(self, labels):
        return [x.value for x in labels]


class MultiLabelFactory(LabelFactory):
    @property
    def empty_label(self):
        return []

    def make_label(self, labels):
        return labels


factories = {
    'str': StrLabelFactory(),
    'strs': StrsLabelFactory(),
    'labels': MultiLabelFactory()
}
