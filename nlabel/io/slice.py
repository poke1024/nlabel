def _slice_i_n(args):
    if not isinstance(args, (list, tuple)):
        raise ValueError(args)
    if len(args) != 2:
        raise ValueError(f"slice filter should be (i, n), is {args}")
    else:
        i, n = map(int, args)
        if n < 2:
            raise ValueError(f"n must be at least 2, is {n}")
        if i < 1 or i > n:
            raise ValueError(f"i must be between 1 and {n}")
        return i, n


class Slice:
    def __init__(self, spec):
        if spec is None:
            self._i = 1
            self._n = 1
        elif isinstance(spec, str):
            self._i, self._n = _slice_i_n(spec.split("/"))
        else:
            self._i, self._n = _slice_i_n(spec)

    def __call__(self, i):
        return 1 + (i % self._n) == self._i
