import collections
import logging
import json
import yaml
from .common import Name
from .form import TagForm
from .parser import TagNameParser


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


class AutoSelector:
    _default_types = {
        'morph': 'strs',
        'feats': 'strs'
    }

    @staticmethod
    def _default_label_type(tagger):
        return OneSelector._default_types.get(tagger, 'str')

    def __init__(self, label_factories):
        self._label_factories = label_factories

    def _make_name(self, tagger_index, tagger, tag_name):
        raise NotImplementedError()

    def build(self, taggers, add):
        tag_forms = {}
        clashes = collections.defaultdict(list)

        for tagger_index, tagger in enumerate(taggers):
            for tag_name, tag_data in tagger['tags'].items():
                name = self._make_name(tagger_index, tagger, tag_name)

                clash = clashes[name.external]
                clash.append(tagger)
                if len(clash) > 1:
                    raise ValueError(
                        f"multiple taggers produce tag '{name.external}: "
                        f"{[x['tagger'] for x in clash]}'")

                label_factory = self._label_factories.get(
                    self._default_label_type(name.internal))

                form = TagForm(name, label_factory)
                assert name.external not in tag_forms
                tag_forms[name.external] = form

                add(tagger_index, form, tag_data)

        return tag_forms


class AllSelector(AutoSelector):
    def _make_name(self, tagger_index, tagger, tag_name):
        return Name(tag_name, f'{tag_name}.{1 + tagger_index}')


class OneSelector(AutoSelector):
    def _make_name(self, tagger_index, tagger, tag_name):
        return Name(tag_name)


def _format_pattern(x, indent='  '):
    return "\n".join([indent + x for x in yaml.dump(x).split("\n")]).rstrip()


def dup_info(pattern, matched):
    s = ["please modify your pattern to match exactly one tagger:"]
    s.append("")
    s.append("pattern:")
    s.append(_format_pattern(pattern))
    for tagger_index, selector, tagger in matched:
        s.append("")
        s.append(f"tagger {tagger_index}:")
        s.append(_format_pattern(tagger['tagger']))
    return "\n".join(s)


def select_taggers(taggers, selector):
    if not isinstance(selector, dict):
        raise ValueError(f'expected dict, got {selector}')
    tagger_selector = TaggerSelector(
        _expand_selector_all(selector))
    for x in taggers:
        if tagger_selector.match_tagger(x._.data['tagger']):
            yield x


class ConfiguredSelector:
    def __init__(self, label_factories, selectors):
        self._nlp_selectors = []
        self._label_factories = {}

        parser = TagNameParser()
        allowed_keys = {'tagger', 'tags'}

        for selector in selectors:
            if selector is None:
                continue

            if not isinstance(selector, dict):
                raise ValueError(f'selector should be a dict, got {selector}')

            x_selector = _expand_selector(selector, 'tagger')

            wrong_keys = set(x_selector.keys()) - allowed_keys
            if wrong_keys:
                raise RuntimeError(f"illegal key '{list(wrong_keys)[0]}' in pattern {selector}")

            tagger_selector = TaggerSelector(x_selector.get('tagger', {}))

            if 'tags' not in x_selector:
                raise RuntimeError(f'expected "tags" inside selector {selector}')

            tag_selectors = []
            for tag in x_selector['tags']:
                tag_data = parser(tag)
                name = tag_data['name']
                tag_selectors.append(name)

                label_type = tag_data['label_type']
                label_factory = label_factories.get(label_type)
                if label_factory is None:
                    raise ValueError(f"unsupported label type {label_type}")

                x_label_factory = self._label_factories.get(name.external)
                if x_label_factory:
                    if x_label_factory is not label_factory:
                        raise ValueError(f"inconsistent label spec for tag '{name.external}'")
                else:
                    self._label_factories[name.external] = label_factory

            self._nlp_selectors.append({
                'tagger': tagger_selector,
                'tags': tag_selectors
            })

    def _add_matched(self, tag_forms, add, tagger_index, selector, tagger):
        tagger_tags = tagger['tags']
        for tag_name in selector['tags']:
            tags_data = tagger_tags.get(tag_name.internal)
            if tags_data is not None:
                form = TagForm(tag_name, self._label_factories[tag_name.external])
                tag_forms[tag_name.external] = form
                add(tagger_index, form, tags_data)

    def build(self, taggers, add):
        tag_forms = {}

        for i, selector in enumerate(self._nlp_selectors):
            matched = []

            for tagger_index, tagger in enumerate(taggers):
                if selector['tagger'].match_tagger(tagger['tagger']):
                    matched.append((tagger_index, selector, tagger))

            if len(matched) > 1:
                logging.info(dup_info(selector['tagger'].pattern, matched))
                raise RuntimeError("multiple taggers are suitable for the selector")

            if not matched:
                for x in self._nlp_selectors:
                    logging.info(json.dumps(x['tagger'].pattern, indent=4))
                raise RuntimeError(
                    f"unmatched tagger selector {selector['tagger'].pattern}")

            self._add_matched(tag_forms, add, *matched[0])

        return tag_forms


class One:
    pass


class All:
    pass


class Automatic:
    pass


def make_selector(label_factories, selectors):
    if not selectors:
        raise ValueError("please specify one or more selectors")
    if len(selectors) == 1 and isinstance(selectors[0], All):
        return AllSelector(label_factories)
    elif len(selectors) == 1 and isinstance(selectors[0], One):
        return OneSelector(label_factories)
    else:
        return ConfiguredSelector(label_factories, selectors)
