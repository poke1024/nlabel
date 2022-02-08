import collections
from .form import TagForm


def match_pattern(pattern, data):
    if isinstance(pattern, dict):
        if not isinstance(data, dict):
            return False
        for k, v in pattern.items():
            data_v = data.get(k)
            if data_v is not None:
                if not match_pattern(v, data_v):
                    return False
            else:
                return False
        return True
    else:
        return pattern == data


def _resolve_pattern(x, k, v):
    ks = k.split('.')
    if len(ks) > 1:
        k0 = ks[0]
        if k0 not in x:
            x[k0] = {}
        _resolve_pattern(x[k0], '.'.join(ks[1:]), v)
    else:
        x[k] = v


def _expand_selector(selector, name):
    prefix = f'{name}.'
    r = collections.defaultdict(dict)
    for k, v in selector.items():
        if k.startswith(prefix):
            _resolve_pattern(r[name], k[len(prefix):], v)
        else:
            r[k] = v
    return r


def _expand_selector_all(selector):
    r = collections.defaultdict(dict)
    for k, v in selector.items():
        parts = k.split('.')
        if len(parts) > 1:
            _resolve_pattern(
                r[parts[0]], '.'.join(parts[1:]), v)
        else:
            r[k] = v
    return r


class TaggerSelector:
    def __init__(self, pattern):
        self._pattern = pattern

    @property
    def pattern(self):
        return self._pattern

    def match_tagger(self, data):
        return match_pattern(self._pattern, data)


def select_taggers(taggers, selector):
    if not isinstance(selector, dict):
        raise ValueError(f'expected dict, got {selector}')
    tagger_selector = TaggerSelector(
        _expand_selector_all(selector))
    for x in taggers:
        if tagger_selector.match_tagger(x._.data['tagger']):
            yield x


class Selector:
    def __init__(self, tags):
        self._by_guid = collections.defaultdict(list)
        name_clashes = collections.defaultdict(list)

        for tag in tags:
            name_clashes[tag._name.external].append(tag)

        for name, clashes in name_clashes.items():
            if len(clashes) > 1:
                raise RuntimeError(f"name clash on {name}")

        for tag in tags:
            self._by_guid[tag.tagger.id].append(tag)

    def build(self, taggers, add):
        tag_forms = {}

        for tagger_index, tagger in enumerate(taggers):
            tags = self._by_guid.get(tagger['guid'], [])
            for tag in tags:
                tag_data = tagger['tags'].get(tag._name.internal)
                if tag_data is not None:
                    form = TagForm(tag)
                    tag_forms[tag._name.external] = form
                    add(tagger_index, form, tag_data)

        return tag_forms


def auto_selectors(selectors, taggers):
    if not selectors:
        name_clashes = collections.defaultdict(list)
        for tagger in taggers:
            for tag in tagger.tags:
                name_clashes[tag.name].append(tag)

        if all(len(v) <= 1 for v in name_clashes.values()):
            selectors = taggers
        else:
            raise RuntimeError(
                f"there are {len(taggers)} taggers with"
                f"conflicting tag names in this archive, "
                f"please use a selector")
    return selectors


class Profile:
    def __init__(self, selectors):
        from nlabel.io.json.group import Tagger, Tag

        self._tags = collections.defaultdict(list)
        taggers = set()

        for x in selectors:
            if isinstance(x, Tagger):
                taggers.add(x)
                self._tags[x].extend(x.tags)
            elif isinstance(x, Tag):
                taggers.add(x.tagger)
                self._tags[x.tagger].append(x)
            else:
                raise ValueError(
                    f"expected Tagger or Tag, got {x}")

        self._taggers = taggers

    @property
    def taggers(self):
        return self._taggers

    def tags(self, tagger):
        return self._tags[tagger]


def make_selector(selectors):
    from nlabel.io.json.group import Tagger, Tag

    tags = []

    for x in selectors:
        if isinstance(x, Tagger):
            tags.extend(x.tags)
        elif isinstance(x, Tag):
            tags.append(x)
        else:
            raise ValueError(
                f"expected Tagger or Tag, got {x}")

    return Selector(tags)
