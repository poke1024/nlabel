from nlabel.io.json.name import Name
from cached_property import cached_property


class TagForm:
    def __init__(self, tag, name=None):
        self._tag = tag
        self._name = tag._name if name is None else name

    def inflections(self):
        yield self
        yield self.pluralize()

    @property
    def tag(self):
        return self._tag

    @property
    def name(self):
        return self._name

    @property
    def is_plural(self):
        return False

    @cached_property
    def label_factory(self):
        return self._tag._label_factory

    def make_empty_label(self, kind):
        return self.label_factory.make_empty_label(kind)

    def make_label(self, data, kind):
        return self.label_factory.make_label(data, kind)

    def pluralize(self):
        return PluralTagForm(self._tag, self._name)


class PluralTagForm(TagForm):
    def __init__(self, tag, name):
        external_name = name.external
        if external_name.endswith('s'):
            plural_name = external_name + '_tags'
        else:
            plural_name = external_name + 's'
        super().__init__(
            tag,
            Name(name.internal, plural_name))
        self._singular_name = name

    @property
    def is_plural(self):
        return True

    @property
    def empty_label(self):
        return []

    def singularize(self):
        return TagForm(
            self._tag, self._singular_name)


def inflected_tag_forms(tag_forms):
    new_tag_forms = {}
    for base_form in tag_forms.values():
        for form in base_form.inflections():
            if form.name.external in new_tag_forms:
                raise ValueError(f'name clash on {form.name.external}')
            new_tag_forms[form.name.external] = form
    return new_tag_forms
