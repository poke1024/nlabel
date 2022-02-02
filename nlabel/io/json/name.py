def normalize_name(s):
    return s.replace('-', '_')


class Name:
    def __init__(self, internal, external=None):
        self._internal = internal
        self._external = external if external else normalize_name(internal)

    @property
    def internal(self):
        return self._internal

    @property
    def external(self):
        return self._external
