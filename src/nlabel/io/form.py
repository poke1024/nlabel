from .common import Name


class TagForm:
    def __init__(self, name, label_factory):
        self._name = name
        self._label_factory = label_factory

    def inflections(self):
        yield self
        yield self.pluralize()

    @property
    def name(self):
        return self._name

    @property
    def is_plural(self):
        return False

    @property
    def label_factory(self):
        return self._label_factory

    @property
    def empty_label(self):
        return self._label_factory.empty_label

    def make_label(self, *args):
        return self._label_factory.make_label(*args)

    def make_label_from_value(self, *args):
        return self._label_factory.make_label_from_value(*args)

    def make_label_from_values(self, *args):
        return self._label_factory.make_label_from_values(*args)

    def pluralize(self):
        return PluralTagForm(
            self._name, self._label_factory)


class PluralTagForm(TagForm):
    def __init__(self, name, label_factory):
        external_name = name.external
        if external_name.endswith('s'):
            plural_name = external_name + '_tags'
        else:
            plural_name = external_name + 's'
        super().__init__(
            Name(name.internal, plural_name),
            label_factory)
        self._singular_name = name

    @property
    def is_plural(self):
        return True

    @property
    def empty_label(self):
        return []

    def singularize(self):
        return TagForm(
            self._singular_name, self._label_factory)


def inflected_tag_forms(tag_forms):
    new_tag_forms = {}
    for base_form in tag_forms.values():
        for form in base_form.inflections():
            if form.name.external in new_tag_forms:
                raise ValueError(f'name clash on {form.name.external}')
            new_tag_forms[form.name.external] = form
    return new_tag_forms
